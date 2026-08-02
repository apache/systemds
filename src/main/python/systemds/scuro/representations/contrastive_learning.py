# -------------------------------------------------------------
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
# -------------------------------------------------------------
import copy


class ContrasitveLearning:
    @staticmethod
    def execute(
        input_first_modality,
        input_second_modality,
        input_first_extensions,
        input_second_extensions,
        metadata_matching_function,
    ):
        # Add check for same dimensionality of input modlities and extensions

        def empty_modality_copy(input_modality):
            modality = copy.deepcopy(input_modality)
            if isinstance(modality, list):
                for m in modality:
                    m.data = []
                    m.metadata = []
            else:
                modality.data = []
                modality.metadata = []

            return modality

        first_modality = empty_modality_copy(input_first_modality)
        second_modality = empty_modality_copy(input_second_modality)
        first_extensions = empty_modality_copy(input_first_extensions)
        second_extensions = empty_modality_copy(input_second_extensions)

        labels = []

        for i in range(len(input_first_modality.data)):
            for j in range(len(input_second_modality.data)):
                first_modality.data.append(input_first_modality.data[i])
                first_modality.metadata.append(input_first_modality.metadata[i])

                for m, input_m in zip(first_extensions, input_first_extensions):
                    m.data.append(input_m.data[i])
                    m.metadata.append(input_m.metadata[i])

                second_modality.data.append(input_second_modality.data[j])
                second_modality.metadata.append(input_second_modality.metadata[j])

                for m, input_m in zip(second_extensions, input_second_extensions):
                    m.data.append(input_m.data[j])
                    m.metadata.append(input_m.metadata[j])

                if metadata_matching_function(
                    input_first_modality.metadata[i], input_second_modality.metadata[j]
                ):
                    labels.append(True)
                else:
                    labels.append(False)

        return (
            [first_modality] + first_extensions,
            [second_modality] + second_extensions,
            labels,
        )
