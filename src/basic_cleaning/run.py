#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Download input artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Loading dataset")
    df = pd.read_csv(artifact_local_path)

    logger.info("Applying basic cleaning")

    # Remove rows with missing prices
    df = df.dropna(subset=["price"])

    # Filter price outliers
    df = df[(df["price"] >= args.min_price) & (df["price"] <= args.max_price)]

    # Filter rows based on latitude and longitude
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save cleaned dataset
    logger.info("Saving cleaned dataset")
    df.to_csv("clean_sample.csv", index=False)

    # Upload artifact to W&B
    logger.info("Uploading artifact")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name and version of the input artifact in W&B",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the artifact that will contain the cleaned dataset",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type assigned to the cleaned artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the cleaned dataset artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum allowed price for filtering outliers",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum allowed price for filtering outliers",
        required=True
    )


    args = parser.parse_args()

    go(args)
