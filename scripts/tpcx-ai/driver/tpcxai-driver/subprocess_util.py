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
# Copyright 2019 Intel Corporation.
# This software and the related documents are Intel copyrighted materials, and your use of them
# is governed by the express license under which they were provided to you ("License"). Unless the
# License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or
# transmit this software or the related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express or implied warranties,
# other than those that are expressly stated in the License.
#
#
import os
import queue
import signal
import subprocess # nosec - the subprocess module is a crucial component to enable the flexibility to add other implementations of this benchmark.
import threading
import time
from pathlib import Path

from .data import SubPhase
from .logger import FileAndDBLogger

state = queue.Queue()


def run_and_capture(command, logger, verbose=False, **kvargs):
    proc_out = {'stdout': [], 'stderr': []}
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
                          universal_newlines=True, **kvargs) as p:
        for line in p.stdout:
            logger.log_out(line)
            proc_out['stdout'].append(line)
            if verbose:
                print(line, end='', flush=True)

    result = subprocess.CompletedProcess(p.args, p.returncode, ''.join(proc_out['stdout']), ''.join(proc_out['stderr']))
    return result


def run_and_log(command, logger, stop_event: threading.Event, verbose=False, **kvargs):
    os.environ['PYTHONUNBUFFERED'] = '1'
    with subprocess.Popen(command, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
                          universal_newlines=True, **kvargs) as p:
        # append process id to global state
        # this is used to kill the process/ process group from a different thread
        state.put(p.pid)
        forbidden_words = ['b\'', '\'', r'\n', r'\r']
        for line in p.stdout:
            for w in forbidden_words:
                line = line.replace(w, '')
            line = line.strip()
            if stop_event.is_set():
                break
            else:
                logger.emit(line)


def run_as_daemon(command, logger, stop_event, verbose=False, **kvargs):
    thread_kvargs = {'command': command.split(), 'logger': logger, 'stop_event': stop_event, 'verbose': verbose,
                     **kvargs}
    daemon_thread = threading.Thread(target=run_and_log, kwargs=thread_kvargs, daemon=True)
    return daemon_thread


def stop_daemons(stop_event):
    stop_event.set()
    while True:
        try:
            pid = state.get(block=False)
            try:
                pg_pid = os.getpgid(pid)
                os.killpg(pg_pid, signal.SIGHUP)
            except ProcessLookupError:
                pass
                # the process was already killed, move on
            state.task_done()
        except queue.Empty:
            break


class Stream(threading.Thread):

    def __init__(self, index, name, actions, db_queue, benchmark_sk, log_file: Path, verboses, tpcxai_home,
                 killall_event, **kvargs):
        if not (len(actions) == len(verboses)):
            raise ValueError(f'lengths of the parameters must match: '
                             f'len(actions)={len(actions)}, len(verboses)={len(verboses)}')
        threading.Thread.__init__(self)
        self.index = index
        self.name = name
        self.actions = actions
        self.db_queue = db_queue
        self.benchmark_sk = benchmark_sk
        self.verboses = verboses
        self.kvargs = kvargs
        self.results = [None] * len(actions)
        self.last_idx = 0
        self.exc = None
        self.killall_event = killall_event
        self.start_times = []
        self.end_times = []
        self.timings = []
        self.log_file = log_file
        self.adabench_home = tpcxai_home
        use_case_sk = self.db_queue.query("SELECT max(command_sk) FROM command").fetchone()[0]
        use_case_sk = use_case_sk + 1 if use_case_sk else 1
        self.base_usecase_sk = use_case_sk

    def run(self) -> None:
        i = 0

        for action, verbose in zip(self.actions, self.verboses):
            if self.killall_event.is_set():
                break
            else:
                msg = []
                if action.subphase.value == SubPhase.INIT.value:
                    msg.append(f"stream {self.name} initializing {action.phase} for uc {action.use_case}")
                elif action.subphase.value == SubPhase.PREPARATION.value:
                    msg.append(f"stream {self.name} preparing {action.phase} for uc {action.use_case}")
                else:
                    msg.append(f"stream {self.name} running {action.phase} for uc {action.use_case}")

                use_case_sk = self.base_usecase_sk + self.index + i
                self.db_queue.insert('''
                                      INSERT INTO command (command_sk, benchmark_fk, use_case, phase, phase_run, sub_phase, command)
                                      VALUES (?, ?, ?, ?, ?, ?, ?)
                                      ''',
                                     (use_case_sk, self.benchmark_sk, action.use_case,
                                      str(action.phase), action.run, str(action.subphase), str(action.command)))
                self.db_queue.insert('INSERT INTO stream (use_case_fk, stream) VALUES (?, ?)', (use_case_sk, self.name))
                log_dir = self.log_file.parent
                action_log_file = log_dir / f"{self.log_file.stem}-{action.phase}-{action.run}-{self.name}-{action.use_case}.out"

                start = time.perf_counter()
                start_time = time.time()
                with FileAndDBLogger(use_case_sk, action_log_file, self.db_queue) as logger:
                    if action.working_dir is not None:
                        result = run_and_capture(action.command, logger, verbose, cwd=action.working_dir, **self.kvargs)
                    else:
                        result = run_and_capture(action.command, logger, verbose, cwd=self.adabench_home,  **self.kvargs)
                end = time.perf_counter()
                end_time = time.time()
                duration = round(end - start, 3)
                self.db_queue.insert('UPDATE command SET return_code = ?, start_time = ?, end_time = ?, runtime = ? WHERE command_sk = ?',
                                     (result.returncode, start_time, end_time, duration, use_case_sk))
                self.start_times.append(start_time)
                self.end_times.append(end_time)
                self.timings.append(duration)
                msg.append(f"in {duration}")
                print('\n'.join(msg))
                self.results[i] = result
                self.last_idx = i
                if result.returncode != 0:
                    self.exc = RuntimeError(f"command {action.command} return {result.returncode}")
                    self.killall_event.set()
                i += 1

    def last_result(self):
        return self.results[self.last_idx]

    def last_action(self):
        return self.actions[self.last_idx]

