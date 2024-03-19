#!/bin/bash

TABLE_FILE="ec2_types.csv"
REGION="us-east-1"

if [ "$#" -eq 1 ]; then
  REGION="$1"
elif [ "$#" -eq 2 ]; then
  REGION="$1"
  TABLE_FILE="$2"
fi

if [ -e "$TABLE_FILE" ]; then
  echo "File $TABLE_FILE exists, do you want to overwrite it?[y/n]"
  read -rp "> " continue
  if [ "$continue" = "y" ]; then
    echo "Overwriting $TABLE_FILE for region '$REGION'."
  else
    exit 0
  fi
else
  echo "Creating $TABLE_FILE for region '$REGION'."
fi

# Create the CSV file ---------------
echo "Fetching list of all supported instance types..."
# header
echo "API_Name,Memory,vCPUs,Family" > "$TABLE_FILE"
# query supported instances and general capabilities
GET_INSTANCE_CMD="aws emr list-supported-instance-types --release-label emr-7.0.0 --region $REGION"
JQ_INFO_QUERY='.SupportedInstanceTypes[] | [.Type, .MemoryGB, .VCPU, .InstanceFamilyId] | @csv'
# $GET_INSTANCE_CMD | jq -r '.SupportedInstanceTypes[] | [.Type, .MemoryGB, .VCPU, .InstanceFamilyId] | @csv' | tr -d '"'
# init call
tmp_output=$($GET_INSTANCE_CMD)
while true; do
  printf "%s" "$(echo $tmp_output | jq -r "$JQ_INFO_QUERY" | tr -d '"')" >> "$TABLE_FILE"
  MARKER=$(echo $tmp_output | jq -r '.Marker')
  if [ "$MARKER" = "null" ]; then
    break
  else
    echo "" >> "$TABLE_FILE"
    tmp_output=$(aws emr list-supported-instance-types --release-label emr-7.0.0 --region $REGION --marker $MARKER)
  fi
done

if [ $? -ne 0 ]; then echo "Command failed."; exit 1; fi
echo "...List (API_Name, Memory, vCPUs) fetched."

echo "Polling the current on-demand price for each supported instance..."
TMP_FILE=$(mktemp)
while IFS=',' read -r col1 rest; do
    # Determine what to append based on the first value (col1)
    case "$col1" in
        "API_Name")
            append="Price"
            ;;
        *)
            PRICE=$(aws pricing get-products --filters "Type=TERM_MATCH,Field=operatingSystem,Value=Linux" \
                    "Type=TERM_MATCH,Field=preInstalledSw,Value=NA" \
                    "Type=TERM_MATCH,Field=regionCode,Value=$REGION" \
                    "Type=TERM_MATCH,Field=capacitystatus,Value=Used" \
                    "Type=TERM_MATCH,Field=tenancy,Value=Shared" \
                    "Type=TERM_MATCH,Field=instanceType,Value=$col1" \
                    --max-results 1 --service-code AmazonEC2 --output text \
                    --query 'PriceList' | jq -r '.terms.OnDemand.[].priceDimensions.[].pricePerUnit.USD')
            # echo "Price for $col1: $PRICE"
            append="$PRICE" # Default action if no specific case matches
            ;;
    esac

    # Append the determined value to the current line and print it
    echo "$col1,$rest,$append" >> "$TMP_FILE"
done < "$TABLE_FILE"

mv "$TMP_FILE" "$TABLE_FILE"