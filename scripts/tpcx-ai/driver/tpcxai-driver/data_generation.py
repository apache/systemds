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

from string import Template

from .data import Phase, SubprocessAction, SubPhase


class DataGeneration:
    _pdgf_init_template = Template("java -jar $pdgf_path -ns")
    _pdgf_training_template = \
        Template("java -Djava.awt.headless=true $pdgf_java_opts -jar $pdgf_path -l $schema -l $generation -ns -sf $scale_factor -sp MY_SEED $seed "
                 "-sp includeLabels 1.0 -sp TTVF 1.0 -s $tables -output '\"$output/\"'")

    _pdgf_training_template_parallel =\
        Template("$tpcxai_home/tools/parallel-data-gen.sh -p $pdgf_path -h nodes -o \"-l $schema -l $generation -ns "
                 "-sf $scale_factor -sp MY_SEED $seed -sp includeLabels 1.0 -sp TTVF 1.0 "
                 "-s $tables -output \\'\\\"$output/\\\"\\'\"")

    _pdgf_serving_template = \
        Template("java -Djava.awt.headless=true $pdgf_java_opts -jar $pdgf_path -l $schema -l $generation -ns "
                 "-sf $scale_factor -sp SF_TRAINING $scale_factor_training -sp MY_SEED $seed "
                 "-sp includeLabels 0.0 -sp TTVF $ttvf -s $tables -output '\"$output/\"'")

    _pdgf_serving_template_parallel = \
        Template("$tpcxai_home/tools/parallel-data-gen.sh -p $pdgf_path -h nodes -o \"-l $schema -l $generation -ns "
                 "-sf $scale_factor -sp SF_TRAINING $scale_factor_training -sp MY_SEED $seed "
                 "-sp includeLabels 0.0 -sp TTVF $ttvf -s $tables -output \\'\\\"$output/\\\"\\'\"")

    _pdgf_scoring_template = \
        Template("java -Djava.awt.headless=true $pdgf_java_opts -jar $pdgf_path -l $schema -l $generation -ns "
                 "-sf $scale_factor -sp SF_TRAINING $scale_factor_training -sp MY_SEED $seed "
                 "-sp includeLabels 2.0 -sp TTVF $ttvf -s $tables -output '\"$output/\"'")

    _pdgf_scoring_template_parallel = \
        Template("$tpcxai_home/tools/parallel-data-gen.sh -p $pdgf_path -h nodes -o \"-l $schema -l $generation -ns "
                 "-sf $scale_factor -sp SF_TRAINING $scale_factor_training -sp MY_SEED $seed "
                 "-sp includeLabels 2.0 -sp TTVF $ttvf -s $tables -output \\'\\\"$output/\\\"\\'\"")

    def __init__(self, seed, tpcxai_home, pdgf_home, pdgf_java_opts, datagen_home, datagen_config, datagen_output, scale_factor=0.1, ttvf=0.01,
                 parallel=False):
        self.seed = seed
        self.tpcxai_home = tpcxai_home
        self.pdgf_home = pdgf_home
        self.pdgf_java_opts = pdgf_java_opts
        self.datagen_home = datagen_home
        self.scale_factor = scale_factor
        self.output = datagen_output
        self.datagen_schema = datagen_config / "tpcxai-schema.xml"
        self.datagen_generation = datagen_config / "tpcxai-generation.xml"
        self.ttvf = ttvf
        self.parallel = parallel
        self.meta_data = {}

    def prepare(self):
        pdgf = self.pdgf_home / 'pdgf.jar'
        cmd = self._pdgf_init_template.substitute(pdgf_path=pdgf)
        return SubprocessAction(0, cmd, Phase.DATA_GENERATION, SubPhase.INIT, metadata=None)

    def run(self, phase, tables):
        if not self.parallel:
            pdgf = self.pdgf_home / 'pdgf.jar'
        else:
            pdgf = self.pdgf_home

        # training
        training_template = self._pdgf_training_template if not self.parallel else self._pdgf_training_template_parallel
        pdgf_train = training_template.substitute(
            pdgf_java_opts=self.pdgf_java_opts,
            tpcxai_home=self.tpcxai_home, pdgf_path=pdgf, seed=self.seed,
            schema=self.datagen_schema, generation=self.datagen_generation,
            scale_factor=self.scale_factor, has_labels=1.0, ttvf=1,
            tables=' '.join(tables), output=self.output / 'training'
        )

        # serving
        serving_template = self._pdgf_serving_template if not self.parallel else self._pdgf_serving_template_parallel
        pdgf_serve = serving_template.substitute(
            pdgf_java_opts=self.pdgf_java_opts,
            tpcxai_home=self.tpcxai_home, pdgf_path=pdgf,
            schema=self.datagen_schema, generation=self.datagen_generation,
            seed=self.seed + 1, scale_factor=self.scale_factor, has_labels=0.0, ttvf=self.ttvf,
            scale_factor_training=self.scale_factor,
            tables=' '.join(tables), output=self.output / 'serving'
        )

        # scoring
        scoring_template = self._pdgf_scoring_template if not self.parallel else self._pdgf_scoring_template_parallel
        pdgf_score_seeded = scoring_template.substitute(
            pdgf_java_opts=self.pdgf_java_opts,
            tpcxai_home=self.tpcxai_home, pdgf_path=pdgf,
            schema=self.datagen_schema, generation=self.datagen_generation,
            seed=self.seed + int(self.scale_factor), scale_factor=1, has_labels=1.0, ttvf=self.ttvf,
            scale_factor_training=self.scale_factor,
            tables=' '.join(tables), output=self.output / 'scoring'
        )

        new_phase = Phase.DATA_GENERATION

        if phase.value == Phase.TRAINING.value:
            cmd = pdgf_train
        elif phase.value == Phase.SERVING.value:
            cmd = pdgf_serve
        elif phase.value == Phase.SCORING.value:
            cmd = pdgf_score_seeded
            new_phase = Phase.SCORING_DATAGEN
        else:
            cmd = ''

        return SubprocessAction(0, cmd, new_phase, SubPhase.WORK, metadata=None)
