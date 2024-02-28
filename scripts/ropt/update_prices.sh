#!/bin/bash

if [ "$#" -eq 2 ]; then
  REGION="$1"
  TABLE_FILE="$2"
else
  echo "Usage: first arg: region code, second arg: file to update"
  exit 0
fi

if [ -e "$TABLE_FILE" ]; then
  echo "Updating prices in $TABLE_FILE for region '$REGION'."
else
  echo "File $TABLE_FILE does not exist."
  exit 1
fi

echo "Polling the current on-demand price for each supported instance..."
TMP_FILE=$(mktemp)
while IFS=',' read -r col1 col2 col3 col4; do
    # Determine what to append based on the first value (col1)
    case "$col4" in
        "Price")
            PRICE="Price"
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
            ;;
    esac

    # Append the determined value to the current line and print it
    echo "$col1,$col2,$col3,$PRICE" >> "$TMP_FILE"
done < "$TABLE_FILE"

mv "$TMP_FILE" "$TABLE_FILE"