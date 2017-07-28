#!/usr/bin/env python3

import gspread
import pprint
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import argparse


# Skip reading 1st row : Header
# Skip reading last row : Footer
def parse_data(file_path):
    csv_file = pd.read_csv(file_path, sep=',', skiprows=1, skipfooter=1, engine='python')
    algo = csv_file['INFO:root:algorithm'].apply(lambda x: x.split(':')[-1])
    key = algo + '_'+ csv_file['run_type'] + '_' + csv_file['intercept'] + '_' + \
                csv_file['matrix_type'] + '_' + csv_file['data_shape']
    return key, csv_file['time_sec']


def auth(path, sheet_name):
    scope = ['https://spreadsheets.google.com/feeds']
    creds = ServiceAccountCredentials.from_json_keyfile_name(path, scope)
    gc = gspread.authorize(creds)
    sheet = gc.open("Perf").worksheet(sheet_name)
    return sheet


def insert_values(sheet, key, col_num, header):
    # Col Name
    sheet.update_cell(1, col_num, header)
    for id, val in enumerate(key):
        sheet.update_cell(id + 2, col_num, val)


def insert_pair(algo, time, start_col, tag):
    insert_values(sheet, algo, start_col, 'algo_{}'.format(tag))
    insert_values(sheet, time, start_col + 1, 'time_{}'.format(tag))
    print('Writing Complete')


def get_dim(sheet):
    try:
        col_count = sheet.get_all_records()
    except:
        col_count = [[]]
    row = len(col_count)
    col = len(col_count[0])
    return row, col

#  ./google_docs.py --file ../temp/test.out --backend singlenode --auth client_json.json --tag 3.0
if __name__ == '__main__':
    execution_mode = ['hybrid_spark', 'singlenode']

    cparser = argparse.ArgumentParser(description='System-ML Update / Stat Script')
    cparser.add_argument('--file', help='Location of the current perf test outputs',
                         required=True, metavar='')
    cparser.add_argument('--backend', help='Backend Type', choices=execution_mode,
                         required=True, metavar='')
    cparser.add_argument('--auth', help='Location to read auth file',
                         required=True, metavar='')
    cparser.add_argument('--tag', help='Tagging header value',
                         required=True, metavar='')

    args = cparser.parse_args()
    arg_dict = vars(args)

    # Authenticate and get sheet dimensions
    sheet = auth(args.auth, args.backend)
    row, col = get_dim(sheet)

    # Read data from file and write to google docs
    algo, time = parse_data(args.file)
    insert_pair(algo, time, col + 1, args.tag)
