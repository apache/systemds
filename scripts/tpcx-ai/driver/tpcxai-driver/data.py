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


from enum import Enum
from functools import total_ordering
from string import Template
from typing import List, Dict, Union


@total_ordering
class Phase(Enum):
    CHECK_INTEGRITY = (-2, -2)
    CLEAN = (-1, 1)
    INIT = (0, 2)
    DATA_GENERATION = (1, 3)
    LOADING = (2, 4)
    TRAINING = (3, 5)
    SERVING = (4, 6)
    SERVING_THROUGHPUT = (5, 6)
    SCORING_DATAGEN = (6, 6)
    SCORING_LOADING = (7, 7)
    SCORING = (8, 8)
    VERIFICATION = (9, 9)

    def __lt__(self, other):
        return self.value < other.value


@total_ordering
class SubPhase(Enum):
    NONE = -1
    INIT = 0
    PREPARATION = 1
    WORK = 2
    POSTWORK = 3
    CLEANUP = 4

    def __lt__(self, other):
        return self.value < other.value


class Action:

    def __init__(self, use_case: int, phase: Phase,
                 sub_phase: SubPhase = SubPhase.NONE, metadata=None, run=1):
        self.use_case = use_case
        self.phase = phase
        self.subphase = sub_phase
        self.duration = None
        self.metadata = metadata
        self.run = run


class SubprocessAction(Action):

    def __init__(self, use_case: int, command: str, phase: Phase, sub_phase: SubPhase = SubPhase.NONE,
                 working_dir=None, metadata=None):
        super().__init__(use_case, phase, sub_phase, metadata)
        self.command = command
        self.working_dir = working_dir


class DaemonAction(SubprocessAction):

    def __init__(self, use_case: int, command: str, phase: Phase, sub_phase: SubPhase = SubPhase.NONE,
                 working_dir=None, metadata=None):
        super().__init__(use_case, command, phase, sub_phase, working_dir, metadata)


class PythonAction(Action):

    def __init__(self, use_case: int, phase: Phase, sub_phase: SubPhase = SubPhase.NONE, metadata=None):
        super().__init__(use_case, phase, sub_phase, metadata)


class ScoringAction(PythonAction):

    def __init__(self, use_case: int, scoring_params: Dict, phase: Phase, sub_phase: SubPhase = SubPhase.NONE,
                 metadata=None):
        super().__init__(use_case, phase, sub_phase, metadata)
        self.scoring_params = scoring_params
        self.verification: Union[None, VerificationAction] = None

    def add_verification(self, metric_name: str, metric_threshold: float, metric_higher_is_better: bool):
        self.verification = VerificationAction(self.use_case, metric_name, metric_threshold, metric_higher_is_better,
                                               Phase.VERIFICATION, SubPhase.WORK, self.metadata)


class VerificationAction(PythonAction):

    def __init__(self, use_case: int, metric_name: str, metric_threshold: float, metric_larger_is_better: bool,
                 phase: Phase, sub_phase: SubPhase = SubPhase.NONE, metadata=None):
        super().__init__(use_case, phase, sub_phase, metadata)
        self.metric_name = metric_name
        self.metric_threshold = metric_threshold
        self.metric_larger_is_better = metric_larger_is_better
        self.metric = None

    def add_metric(self, metric: float):
        self.metric = metric

    def meets_quality_threshold(self):
        if self.metric:
            if self.metric_larger_is_better:
                return self.metric >= self.metric_threshold
            else:
                return self.metric <= self.metric_threshold
        else:
            raise RuntimeError('No metric has been set. Please set a metric with `add_metric`.')

    def get_metric_command(self):
        comparator = '>=' if self.metric_larger_is_better else '<='
        return f"{self.metric} {comparator} {self.metric_threshold}"


class DataStore:

    def __init__(self, name: str, create_template: Template, copy_template: Template,
                 load_template: Template, load_dir_template: Template,
                 delete_template: Template, download_template: Template, delete_parallel_template: Template = None):
        self.name = name
        self.create = create_template
        self.copy = copy_template
        self.load = load_template
        self.load_dir = load_dir_template
        self.delete = delete_template
        self.download = download_template
        self.delete_parallel = delete_parallel_template


def datastore_from_dict(dictionary: dict):
    load_dir = dictionary['load_directory'] if 'load_directory' in dictionary else dictionary['load']
    load_dir = Template(load_dir)
    if 'delete_parallel' in dictionary:
        return DataStore(dictionary['name'],
                         Template(dictionary['create']), Template(dictionary['copy']),
                         Template(dictionary['load']), load_dir,
                         Template(dictionary['delete']), Template(dictionary['download']),
                         Template(dictionary['delete_parallel']))
    else:
        return DataStore(dictionary['name'],
                         Template(dictionary['create']), Template(dictionary['copy']),
                         Template(dictionary['load']), load_dir,
                         Template(dictionary['delete']), Template(dictionary['download']))


class Metadata:

    def __init__(self, num_records):
        self.num_records = num_records
