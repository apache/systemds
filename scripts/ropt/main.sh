#!/bin/bash

# Load common functions
source common.sh

# Load the manually set configs
source systemds_cluster.config

#Create systemDS buckets in S3
#LocationConstraint configuration required regions outside of us-east-1
#if [ "$REGION" = "us-east-1" ]; then LOCATION_CONSTRAINT=""; else LOCATION_CONSTRAINT="--create-bucket-configuration LocationConstraint=$REGION"; fi
#aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" "$LOCATION_CONSTRAINT"
#aws s3api create-bucket --bucket "$BUCKET-logs" --region "$REGION" "$LOCATION_CONSTRAINT"
#
## Upload SystemDS.jar
#aws s3 cp "$SYSTEMDS_JAR" "s3://$BUCKET"
## Upload DML file
#aws s3 cp "$DML_FILE" "s3://$BUCKET"

# Run the resource optimization program
java -jar /Users/lachezarnikolov/my_projects/thesis/systemds/target/systemds-3.3.0-SNAPSHOT-ropt.jar

# Output the computed optimal cluster configurations
echo "Check carefully the generated optimal configuration:"
jq "." "./instances.json"
# TODO: Print the a price per hour
read -rp "[y/n]: " continue
if [ "$continue" = "y" ]; then
  echo "A cluster with this configurations will be launched in region '$REGION'."
else
  exit 0
fi

# Launch cluster
