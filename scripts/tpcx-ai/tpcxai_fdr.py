#!/usr/bin/env python

#
# Copyright (C) 2021 Transaction Processing Performance Council (TPC) and/or its contributors.
# This file is part of a software package distributed by the TPC
# The contents of this file have been developed by the TPC, and/or have been licensed to the TPC under one or more contributor
# license agreements.
# This file is subject to the terms and conditions outlined in the End-User
# License Agreement (EULA) which can be found in this distribution (EULA.txt) and is available at the following URL:
# http://www.tpc.org/TPC_Documents_Current_Versions/txt/EULA.txt
# Unless required by applicable law or agreed to in writing, this software is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, and the user bears the entire risk as to quality
# and performance as well as the entire cost of service or repair in case of defect. See the EULA for more details.
#


#
# Copyright 2021 Intel Corporation.
# This software and the related documents are Intel copyrighted materials, and your use of them 
# is governed by the express license under which they were provided to you ("License"). Unless the 
# License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
# transmit this software or the related documents without Intel's prior written permission.
# 
# This software and the related documents are provided as is, with no express or implied warranties, 
# other than those that are expressly stated in the License.
# 
#


import argparse
import base64
import io
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path,PurePath

from datetime import timedelta
import jinja2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def parse_args(args=None):
    if not args:
        args = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--database', metavar='FILE', type=str, required=True)
    parser.add_argument('-b', '--benchmark', metavar='ID', type=int, required=False)
    parser.add_argument('-f', '--file', metavar='OUTPUT FILE', type=str, required=False)
    parser.add_argument('-t', '--type', metavar='TYPE OF REPORT', choices=['txt', 'html', 'json'], default='txt')
    parser.add_argument('--include-clean', action='store_true', default=False)

    args = parser.parse_args(args[1:])
    if check_args(args) ==False:
       return None
    return args

def check_args(args):
    db_path = Path(args.database)
    try:
        db_path.resolve().relative_to(Path().resolve().parent)
        if db_path.exists()==False:
            print(args.database, 'is not a valid path')
            return False
    except:
        print(args.database, 'is not a valid path')
        return False
    ##
    out_path=Path(args.file)
    try:
        out_path.resolve().relative_to(Path().resolve().parent)
    except:
        print(args.file, 'is not a valid path inside the benchmark top level directory')
        return False

def get_benchmark_id(connection: sqlite3.Connection, benchmark_id=None):
    """
    Returns given benchmark_id if exists or the id of the last benchmark run
    :param connection: Database connection to the tpcxai.db file
    :param benchmark_id: ID of the benchmark to check or None
    :return:
    """
    if benchmark_id:
        return benchmark_id
    else:
        query = 'SELECT benchmark_sk FROM benchmark ORDER BY start_time DESC LIMIT 1'
        result = connection.execute(query)
        return result.fetchone()[0]


def format_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).isoformat(sep=' ', timespec='milliseconds')


def rename_stream(stream, phase):
    if phase in ['Phase.DATA_GENERATION', 'Phase.SCORING_DATAGEN']:
        new_stream = 'DATA_GENERATION'
    elif phase in ['Phase.CLEAN']:
        new_stream = 'CLEAN'
    elif phase in ['Phase.LOADING']:
        new_stream = 'LOAD_TEST'
    elif phase in ['Phase.SCORING', 'Phase.VERIFICATION', 'Phase.SCORING_LOADING']:
        new_stream = 'SCORING'
    else:
        new_stream = stream

    return new_stream


def make_report(connection: sqlite3.Connection, benchmark_id, include_clean=True):
    report = {'benchmark_id': benchmark_id}

    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    meta_query = '''
        SELECT benchmark_name, start_time, end_time, cmd_flags, successful 
        FROM benchmark 
        WHERE benchmark_sk = ?
    '''
    result = cursor.execute(meta_query, (benchmark_id,))
    benchmark_meta = result.fetchone()
    report['benchmark_name'] = benchmark_meta['benchmark_name']
    report['start'] = benchmark_meta['start_time']
    report['end'] = benchmark_meta['end_time']
    if report['start'] and report['end']:
        report['duration'] = "{}".format(str(timedelta(seconds=round(benchmark_meta['end_time'] - benchmark_meta['start_time'],3))))
    else:
        report['duration'] = -1
    report['cmd_args'] = benchmark_meta['cmd_flags']
    report['successful'] = benchmark_meta['successful']
    # detailed failure object
    report['failures'] = []
    # work around for old benchmark runs where not all return_codes are reported
    # that means for some command the return_code is NULL which does not indicate failure but success
    if not report['successful']:
        success_query = 'SELECT total(return_code) AS successful FROM command WHERE benchmark_fk = ?'
        result = cursor.execute(success_query, (benchmark_id, )).fetchone()
        report['successful'] = True if result['successful'] == 0 else False

    if not bool(report['successful']):
        qry = '''
            SELECT command_sk, use_case, phase, sub_phase, command, return_code, group_concat(log, '\n') AS log
            FROM command LEFT JOIN log_std_out ON command_sk = use_case_fk
            WHERE return_code != 0 AND benchmark_fk = ?
            GROUP BY command_sk
        '''
        result = cursor.execute(qry, (benchmark_id, ))
        for rec in result.fetchall():
            failure = {'use_case': rec['use_case'], 'phase': rec['phase'], 'sub_phase': rec['sub_phase'],
                       'command': rec['command'], 'return_code': rec['return_code'], 'log': rec['log']}
            report['failures'].append(failure)

    metric_query = 'SELECT metric_name, metric_value ' \
                   'FROM performance_metric ' \
                   'WHERE benchmark_fk = ? ' \
                   'ORDER BY metric_time'
    result = cursor.execute(metric_query, (benchmark_id, ))
    metrics = []
    for rec in result.fetchall():
        metrics.append(dict(rec))
    report['metric'] = metrics

    phase_runtime_query = '''
        SELECT stream, phase, phase_run,
               min(start_time) AS start_time, max(end_time) AS end_time,
               sum(runtime) AS runtime, CASE sum(return_code) WHEN 0 THEN 'True' ELSE 'False' END successful
        FROM command JOIN stream ON command_sk = use_case_fk
        WHERE benchmark_fk = ? AND
            (sub_phase = 'SubPhase.WORK' OR (phase = 'Phase.DATA_GENERATION' AND sub_phase = 'SubPhase.NONE'))
        GROUP BY stream, phase, phase_run
        ORDER BY start_time
    '''
    phases_with_runs = cursor.execute(
        'SELECT phase, COUNT(DISTINCT phase_run) AS runs FROM command WHERE benchmark_fk = ? GROUP BY phase HAVING runs > 1',
        (benchmark_id, )
    ).fetchall()
    phases_with_runs = list(map(lambda r: r['phase'], phases_with_runs))
    result = cursor.execute(phase_runtime_query, (benchmark_id,))
    phases = []
    for rec in result.fetchall():
        phase_name = rec['phase']
        phase_name_formatted = f"{rec['phase']}_{rec['phase_run']}" if rec['phase'] in phases_with_runs else rec['phase']
        phase_name_formatted = phase_name_formatted.replace('Phase.', '')
        phase_use_case_query = '''
            SELECT command.use_case, command.start_time, command.end_time, command.runtime,
                   command.command,
                   quality_metric.metric_name, printf("%.5f",quality_metric.metric_value) as metric_value, stream.stream,
                   return_code
            FROM (command
                JOIN stream ON command.command_sk = stream.use_case_fk)
                LEFT JOIN quality_metric ON command.command_sk = quality_metric.use_case_fk
            WHERE benchmark_fk = ? AND phase = ? AND sub_phase = 'SubPhase.WORK' AND stream = ? AND phase_run = ?
            ORDER BY command.start_time
        '''
        uc_result = cursor.execute(phase_use_case_query, (benchmark_id, phase_name, rec['stream'], rec['phase_run']))
        use_cases = []
        for uc_rec in uc_result.fetchall():
            start_time = uc_rec['start_time']
            end_time = uc_rec['end_time']
            uc = {'use_case': uc_rec['use_case'],
                  'start_time': start_time, 'end_time': end_time, 'runtime': uc_rec['runtime'], 'command': uc_rec['command'],
                  'metric_name': uc_rec['metric_name'], 'metric_value': uc_rec['metric_value'],
                  'stream': rename_stream(rec['stream'], phase_name),
                  'return_code': uc_rec['return_code'], 'successful': True if uc_rec['return_code'] == 0 else False}
            use_cases.append(uc)
        phase = {'stream': rename_stream(rec['stream'], phase_name),
                 'phase': phase_name, 'phase_run': rec['phase_run'], 'phase_name_formatted': phase_name_formatted,
                 'runtime': round(rec['runtime'], 3) if rec['runtime'] else rec['runtime'],
                 'start_time': rec['start_time'], 'end_time': rec['end_time'],
                 'successful': rec['successful'], 'use_cases': use_cases}
        # add all phase except for CLEAN, which is only added when include_clean is True
        # this means CLEAN is only add if explicitly specified
        if not phase['phase'].startswith('Phase.CLEAN') or include_clean:
            phases.append(phase)

    report['phases'] = phases

    # per use-case table
    use_case_res = cursor.execute("SELECT * FROM command WHERE benchmark_fk = ? AND sub_phase = 'SubPhase.WORK'",
                                  (benchmark_id, ))
    records = []
    for rec in use_case_res.fetchall():
        records.append(dict(rec))

    df = pd.DataFrame.from_records(records)
    df['phase_name'] = np.where(df['phase'].isin(phases_with_runs),
                                df['phase'] + '_' + df['phase_run'].astype(str),
                                df['phase'])
    df['phase_name'] = df['phase_name'].str.replace('Phase.', '', regex=False)
    df['phase_name'] = df['phase_name'].str.replace('_nan', '').str.replace('.0', '', regex=False)
    phase_sorting = df.groupby(['phase_name'])['start_time'].min().sort_values().index.values
    df_mean = df.pivot_table(values='runtime', index='use_case', columns='phase_name').reset_index()
    df_sum = df.pivot_table(values='runtime', index='use_case', columns='phase_name', aggfunc='sum').reset_index()
    df = df_sum
    if 'SERVING_THROUGHPUT' in df.columns:
        df['SERVING_THROUGHPUT'] = df_mean['SERVING_THROUGHPUT']
    new_index = list(set(df.columns) - set(phase_sorting)) + list(phase_sorting)
    df = df.reindex(new_index, axis='columns')
    df = df.round({'CLEAN': 3, 'DATA_GENERATION': 3, 'LOADING': 3, 'TRAINING': 3, 'SCORING_DATAGEN': 3, 'SERVING_1': 3,
                   'SERVING_2': 3, 'SCORING_LOADING': 3, 'SCORING': 3, 'SERVING_THROUGHPUT': 3})
    df = df.rename(columns={'SERVING_THROUGHPUT': 'SERVING_THROUGHPUT (AVG)'})
    df = df.drop(columns='VERIFICATION')
    report['use_cases'] = df.to_dict(orient='records')

    return report


def write_report_json(report, path=None):
    opened = False
    if not path:
        file = sys.stdout
    elif isinstance(path, (str, Path)):
        file = open(path, 'w')
        opened = True
    else:
        file = path

    json.dump(report, file, indent=4)
    file.write('\n')
    if opened:
        file.close()


def get_table(phases, phase, phase_name=None, phase_run=None, aggregate_only=False, format_datetime=False):
    """
    Create dataframe of the form
             Stream            Phase  Use Case     Runtime Successful         Comment
         POWER_TEST  DATA_GENERATION       0.0  192.166866       True  complete phase
         POWER_TEST  DATA_GENERATION       0.0  119.355436       True
         POWER_TEST  DATA_GENERATION       0.0   72.811429       True
    :param phases:
    :param phase:
    :param phase_name:
    :param phase_run:
    :param aggregate_only:
    :return:
    """
    if not phase_name:
        phase_name = phase

    if phase_run:
        filtered_phases = list(filter(lambda p: p['phase'] == phase and p['phase_run'] == phase_run, phases))
    else:
        filtered_phases = list(filter(lambda p: p['phase'] == phase, phases))
    table = pd.DataFrame({'Stream': [], 'Phase': [], 'Use Case': [], 'Start Time': [], 'End Time': [], 'Runtime': [],
                          'Successful': [], 'Comment': []})
    table['Use Case'] = table['Use Case'].astype(int)
    for contents in filtered_phases:
        sts = []
        ps = []
        pr = []
        ucs = []
        ts = []
        ss = []
        cs = []
        st = []
        et = []
        sts.append(contents['stream'])
        phase_name_formatted = contents['phase_name_formatted']
        ps.append(phase_name_formatted)
        # pr.append(contents['phase_run'])
        ucs.append(0)
        ts.append(contents['runtime'])
        ss.append(contents['successful'])
        cs.append('complete phase')
        if format_datetime:
            st.append(format_time(contents['start_time']))
            et.append(format_time(contents['end_time']))
        else:
            st.append(contents['start_time'])
            et.append(contents['end_time'])
        if not aggregate_only:
            for uc in contents['use_cases']:
                sts.append(contents['stream'])
                ps.append(phase_name_formatted)
                pr.append(contents['phase_run'])
                ucs.append(uc['use_case'])
                ts.append(uc['runtime'])
                ss.append(uc['successful'])
                if phase == 'Phase.SCORING':
                    cs.append(f"{uc['metric_name']}: {uc['metric_value']}")
                elif phase in ['Phase.CLEAN', 'Phase.LOADING']:
                    cs.append(uc['command'])
                else:
                    cs.append('')
                if format_datetime:
                    st.append(format_time(uc['start_time']))
                    et.append(format_time(uc['end_time']))
                else:
                    st.append(uc['start_time'])
                    et.append(uc['end_time'])
        t = pd.DataFrame({'Stream': sts, 'Phase': ps, 'Use Case': ucs,
                          'Start Time': st, 'End Time': et, 'Runtime': ts,
                          'Successful': ss, 'Comment': cs})
        table = table.append(t)
    return table


def output_image(path=None, ax=None, format='png'):
    img = None
    if path:
        if ax:
            ax.figure.savefig(path, bbox_inches='tight')
        else:
            plt.savefig(path, format=format, bbox_inches='tight')
    else:
        img_bytes = io.BytesIO()
        if ax:
            ax.figure.savefig(img_bytes, format=format, bbox_inches='tight')
        else:
            plt.savefig(img_bytes, format=format, bbox_inches='tight')
        # img_bytes.seek(0)
        img = base64.b64encode(img_bytes.getvalue()).decode("utf-8").replace("\n", "")

    return img


def make_graphs(report, output=None):
    label_mapping = {1: '1_tpc',
                     2: '2_tpc',
                     3: '3_tpc',
                     4: '4_tpc',
                     5: '5_tpc',
                     6: '6_tpc',
                     7: "7_tpc",
                     8: "8_tpc",
                     9: "9_tpc",
                     10: '10_tpc',
                     11: "1_sds",
                     12: "2:sds",
                     13: "3_sds",
                     14: "4_sds",
                     15: "5_sds",
                     16: "6_sds",
                     17: "7_sds",
                     18: "8_sds",
                     19: "9_sds",
                     20: "10_sds"}

    result = {}
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # phases
    phases_runtimes = [{
        'phase': p['phase'].replace('Phase.', ''),
        'phase_name_formatted': p['phase_name_formatted'],
        'runtime': p['runtime']
    } for p in report['phases']]
    phases_runtimes = pd.DataFrame(phases_runtimes)
    use_cases_df = pd.DataFrame.from_records(report['use_cases'])

    # pie chart
    figsize = (32, 8)
    figure = plt.figure(1, (32, 8))
    plt.rcParams.update({'font.size': 22})
    plt.pie(phases_runtimes['runtime'], labels=phases_runtimes['phase_name_formatted'], autopct='%1.1f%%')
    if output:
        img = f"{output}/pie.png"
        output_image(path=img)
    else:
        img = output_image()
    result['phases_pie'] = img

    # bar chart
    ax = phases_runtimes.plot.bar(x='phase_name_formatted', y='runtime', color=color_cycle, figsize=figsize, rot=0)
    ax.set_ylabel('runtime (s)')
    ax.legend('')
    if output:
        img = f"{output}/phases.png"
        output_image(path=img)
    else:
        img = output_image(ax=ax)
    result['phases_bar'] = img

    ax = phases_runtimes.plot.bar(x='phase_name_formatted', y='runtime', color=color_cycle, logy=True, figsize=figsize, rot=0)
    ax.set_ylabel('runtime (s)')
    ax.legend('')
    if output:
        img = f"{output}/phases_log.png"
        output_image(path=img)
    else:
        img = output_image(ax=ax)
    result['phases_bar_log'] = img

    # training
    lst = [ucs['use_cases'] for ucs in report['phases'] if ucs['phase'] == 'Phase.TRAINING']
    if len(lst) > 0:

        lst = lst[0]
        lst = list(map(lambda uc: {'use_case': uc['use_case'], 'runtime': uc['runtime']}, lst))
        training_runtimes = pd.DataFrame(lst)

        training_runtimes['use_case'] = training_runtimes['use_case'].map(label_mapping)

        # bar chart
        ax = training_runtimes.plot.bar(x='use_case', y='runtime', color=color_cycle, figsize=figsize, logy=True, rot=0)
        ax.set_ylabel('runtime (s)')
        ax.legend('')
        if output:
            img = f"{output}/training_bar.png"
            output_image(path=img)
        else:
            img = output_image(ax=ax, format='svg')
        result['training_bar'] = img

    # serving
   # new_order = [0, 4, 1, 5, 2, 6, 3, 7]
    serving_times = use_cases_df[use_cases_df['use_case'] != 0]
    #serving_times = serving_times.iloc[new_order].reset_index(drop=True)
    serving_times['use_case'] = serving_times['use_case'].map(label_mapping)
    serving_times = serving_times.set_index('use_case')
    serving_cols = [col for col in serving_times.columns if col.startswith("SERVING")]
    serving_times = serving_times[serving_cols]

    if len(serving_times) > 0:
        # bar chart
        ax = serving_times.plot.bar(color=color_cycle, figsize=figsize, rot=0)
        ax.set_ylabel('runtime (s)')
        # ax.legend('')
        if output:
            img = f"{output}/serving_bar_grouped.png"
            output_image(path=img)
        else:
            img = output_image(ax=ax, format='svg')
        result['serving_bar_grouped'] = img

    # serving throughput error bar
    serving_throughput_times = []
    for phase in report['phases']:
        if phase['phase_name_formatted'] == 'SERVING_THROUGHPUT':
            stream = phase['stream']
            for use_case in phase['use_cases']:
                rec = {
                    'use_case': use_case['use_case'],
                    'stream': stream,
                    'runtime': use_case['runtime']
                }
                serving_throughput_times.append(rec)

    serving_throughput_times = pd.DataFrame.from_records(serving_throughput_times)
    #new_order = [0, 4, 1, 5, 2, 6, 3, 7]

    serving_throughput_times['use_case'] = serving_throughput_times['use_case'].map(label_mapping)
    serving_throughput_times = pd.pivot_table(serving_throughput_times,
                                              values='runtime', index='stream', columns='use_case')
    if len(serving_throughput_times) > 0:
        ax = serving_throughput_times.plot.box(figsize=figsize, rot=0)
        ax.set_ylabel('runtime (s)')
        if output:
            img = f"{output}/serving_throughput_error.png"
            output_image(path=img)
        else:
            img = output_image(ax=ax, format='svg')
        result['serving_throughput_error'] = img

    # use_cases
    use_cases_phases = use_cases_df[use_cases_df['use_case'] != 0]
    #use_cases_phases = use_cases_phases.iloc[new_order].reset_index(drop=True)
    use_cases_phases = use_cases_phases.set_index('use_case')
    use_cases_cols = [col for col in use_cases_phases.columns if col.startswith("TRAINING") or col.startswith("SERVING")]
    use_cases_phases = use_cases_phases[use_cases_cols]



    use_cases_phases['use case'] = use_cases_phases.index.map(label_mapping)
    use_cases_phases.set_index('use case', inplace=True)
    if len(use_cases_phases) > 0:
        # bar chart
        ax = use_cases_phases.plot.barh(color=color_cycle, stacked=True, figsize=figsize, rot=0)
        ax.set_xlabel('runtime (s)')
        # ax.legend('')
        if output:
            img = f"{output}/use_cases_stacked.png"
            output_image(path=img)
        else:
            img = output_image(ax=ax, format='svg')
        result['use_cases_stacked'] = img

    return result


def write_report_html(report, path=None, **kwargs):
    opened = False
    if not path:
        file = sys.stdout
    elif isinstance(path, (str, Path)):
        file = open(path, 'w')
        opened = True
    else:
        file = path

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    with open(f'{script_dir}/tpcxai_fdr_template.html') as template_file:
        template = jinja2.Template(template_file.read())
    report = update_use_case_labels(report)
    html = template.render(report, **kwargs)
    file.write(html)
    file.write('\n')
    if opened:
        file.close()

def update_use_case_labels(report):
    label_mapping2 = {1: '1_tpc',
                     2: '2_tpc',
                     3: '3_tpc',
                     4: '4_tpc',
                     5: '5_tpc',
                     6: '6_tpc',
                     7: "7_tpc",
                     8: "8_tpc",
                     9: "9_tpc",
                     10: '10_tpc',
                     11: "1_sds",
                     12: "2:sds",
                     13: "3_sds",
                     14: "4_sds",
                     15: "5_sds",
                     16: "6_sds",
                     17: "7_sds",
                     18: "8_sds",
                     19: "9_sds",
                     20: "10_sds"}

    for phase in report['phases']:
        for uc in phase['use_cases']:
            if uc['use_case'] in label_mapping2:
                uc['use_case'] = label_mapping2[uc['use_case']]
    for uc in report['use_cases']:
        if uc['use_case'] in label_mapping2:
            uc['use_case'] = label_mapping2[uc['use_case']]
    return report

def write_report_txt(report, path=None, exclude_clean=False):
    opened = False
    if not path:
        file = sys.stdout
    elif isinstance(path, (str, Path)):
        file = open(path, 'w')
        opened = True
    else:
        file = path

    perf_metric_tbl = pd.DataFrame.from_records(report['metric'])

    tables = pd.DataFrame({'Stream': [], 'Phase': [], 'Use Case': [], 'Start Time': [], 'End Time': [], 'Runtime': [],
                          'Successful': [], 'Comment': []})
    tables = tables.astype({'Use Case': int})
    phases = report['phases']

    clean_tbl = get_table(phases, 'Phase.CLEAN', 'CLEAN', format_datetime=True)
    tables = tables.append(clean_tbl)
    data_gen_tbl = get_table(phases, 'Phase.DATA_GENERATION', 'DATA_GENERATION', format_datetime=True)
    tables = tables.append(data_gen_tbl)
    loading_tbl = get_table(phases, 'Phase.LOADING', 'LOADING', format_datetime=True)
    tables = tables.append(loading_tbl)
    training_tbl = get_table(phases, 'Phase.TRAINING', 'TRAINING', format_datetime=True)
    tables = tables.append(training_tbl)
    serving_tbl = get_table(phases, 'Phase.SERVING', 'SERVING', format_datetime=True)
    tables = tables.append(serving_tbl)
    scoring_tbl = get_table(phases, 'Phase.SCORING', 'SCORING', format_datetime=True)
    tables = tables.append(scoring_tbl)
    serving_throughput_tbl = get_table(phases, 'Phase.SERVING_THROUGHPUT', 'SERVING_THROUGHPUT', format_datetime=True)
    tables = tables.append(serving_throughput_tbl)

    uc_table = pd.DataFrame.from_records(report['use_cases'])

    file.write(f"BENCHMARK ID: {report['benchmark_id']}\n")
    file.write("\n")
    file.write(f"BENCHMARK START: {format_time(report['start'])}\n")
    file.write(f"BENCHMARK END: {format_time(report['end'])}\n")
    file.write(f"BENCHMARK DURATION: {report['duration']}\n")
    file.write(f"BENCHMARK NAME: {report['benchmark_name']}\n")
    file.write(f"BENCHMARK METRIC:\n")
    file.write(perf_metric_tbl.to_string(index=False, header=True))
    file.write("\n")
    file.write(f"CMD ARGS: {report['cmd_args']}\n")
    file.write("\n")
    file.write(tables.to_string(index=False, header=True))
    file.write("\n\n")
    file.write(uc_table.to_string(index=False, header=True, na_rep=''))
    file.write("\n\n")
    if bool(report['successful']):
        file.write("Benchmark run is valid\n")
    else:
        file.write("Benchmark run is NOT valid\n")
        for failure in report['failures']:
            file.write("FAILURE: ")
            file.write(f"use case {failure['use_case']} failed during phase {failure['phase'], failure['sub_phase']}\n")
            file.write(f"command was: {failure['command']}\n")
            file.write(f"command returned: {failure['return_code']}\n")
            file.write("\n")
            file.write(failure['log'])
            file.write("\n")

    file.write("\n")

    if opened:
        file.close()


def main():
    args = parse_args()
    if args is None:
        return 1

    file = sys.stdout if not args.file else args.file
    include_clean = args.include_clean

    connection = sqlite3.connect(args.database)
    benchmark_id = get_benchmark_id(connection, args.benchmark)

    report = make_report(connection, benchmark_id, include_clean)

    if args.type.lower() == 'json':
        write_report_json(report, file)
    elif args.type.lower() == 'html':
        graphs = make_graphs(report, None)
        write_report_html(report, file, **graphs)
    elif args.type.lower() == 'txt':
        write_report_txt(report, file)
    else:
        print(f"report type {args.type} is invalid", sys.stderr)
        exit(1)


if __name__ == '__main__':
    main()
