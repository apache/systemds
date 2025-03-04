#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

import argparse
import csv
import json
import os
import pandas as pd
import boto3


def update_prices(region: str, table_file: str):
    price_table = dict()

    # init target instance types
    with open(table_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            col1 = row[0]
            if col1 != "API_Name":
                # NaN to indicate types not supported for the target region
                price_table[col1] = pd.NA

    # now get the actual prices from the AWS API
    client = boto3.client('pricing', region_name=region)

    # fetch all products using pagination
    print(f"Fetching current priced for the target instances in region '{region}'...")
    next_token = None
    while True:
        filters = [
            {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
            {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
            {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
            {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
            {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
            {"Type": "TERM_MATCH", "Field": "gpuMemory", "Value": "NA"}
        ]

        if next_token:
            response = client.get_products(ServiceCode="AmazonEC2", Filters=filters, MaxResults=100, NextToken=next_token)
            print("\tanother 100 records have been fetched...")
        else:
            response = client.get_products(ServiceCode="AmazonEC2", Filters=filters, MaxResults=100)
            print("\t100 records have been fetched...")

        # extract the price from the response
        for product in response.get("PriceList", []):
            product_data = json.loads(product)
            instance_type = product_data["product"]["attributes"]["instanceType"]
            # get price only for target instances
            if instance_type in price_table:
                price = next(iter(next(iter(product_data["terms"]["OnDemand"].values()))["priceDimensions"].values()))["pricePerUnit"]["USD"]
                price_table[instance_type] = float(price)

        # handle pagination
        next_token = response.get('NextToken')
        if not next_token:
            break

    print(f"...all prices has been fetched successfully.")
    # update the csv table
    ec2_df = pd.read_csv(table_file)
    for instance_type, price in price_table.items():
        ec2_df.loc[ec2_df["API_Name"] == instance_type, "Price"] = price
    ec2_df.to_csv(table_file, index=False, na_rep="N/A")
    print(f"Prices have been updated to file {table_file}")
    

def main():
    parser = argparse.ArgumentParser(description='Update prices in table with EC2 instance stats')
    parser.add_argument('region', help='Target AWS region (e.g., us-east-1).')
    parser.add_argument('table_file', help='CSV file to be updated')
    
    args = parser.parse_args()

    if not os.path.exists(args.table_file):
        print(f"The given file for update does not exists")
        exit(1)
    # the actual price update logic
    update_prices(args.region, args.table_file)

if __name__ == "__main__":
    main()
