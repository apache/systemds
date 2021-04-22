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

import os
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from py4j.java_gateway import JavaObject, JVMView
from systemds.operator import OperationNode
from systemds.utils.consts import VALID_INPUT_TYPES
from systemds.utils.converters import pandas_to_frame_block
from systemds.script_building.dag import OutputType, DAGNode


class Frame(OperationNode):
    def __init__(
        self,
        sds_context: "SystemDSContext",
        df: pd.DataFrame,
        *args: Sequence[VALID_INPUT_TYPES],
        **kwargs: Dict[str, VALID_INPUT_TYPES]
    ) -> None:

        if args:
            unnamed_params = args
        else:
            unnamed_params = ["'./tmp/{file_name}'"]
            unnamed_params.extend(args)

        named_params = {"data_type": '"frame"'}

        self._pd_dataframe = df
        self.shape = None
        if isinstance(self._pd_dataframe, pd.DataFrame):
            self.shape = self._pd_dataframe.shape
            named_params["rows"] = self.shape[0]
            named_params["cols"] = self.shape[1]

        named_params.update(kwargs)
        super().__init__(
            sds_context,
            "read",
            unnamed_params,
            named_params,
            output_type=OutputType.FRAME,
            is_python_local_data=self._is_pandas(),
            shape=self.shape,
        )

    def pass_python_data_to_prepared_script(
        self, sds, var_name: str, prepared_script: JavaObject
    ) -> None:
        assert (
            self.is_python_local_data
        ), "Can only pass data to prepared script if it is python local!"
        if self._is_pandas():
            prepared_script.setFrame(
                var_name, pandas_to_frame_block(sds, self._pd_dataframe), True
            )  # True for reuse

    def code_line(
        self,
        var_name: str,
        unnamed_input_vars: Sequence[str],
        named_input_vars: Dict[str, str],
    ) -> str:
        code_line = super().code_line(var_name, unnamed_input_vars, named_input_vars)
        if self._is_pandas():
            code_line = code_line.format(file_name=var_name)
        return code_line

    def compute(
        self, verbose: bool = False, lineage: bool = False
    ) -> Union[pd.DataFrame]:
        if self._is_pandas():
            if verbose:
                print("[Pandas Frame - No Compilation necessary]")
            return self._pd_dataframe
        else:
            return super().compute(verbose, lineage)

    def _is_pandas(self) -> bool:
        return self._pd_dataframe is not None

    def transform_encode(self, spec):
        self._check_frame_op()
        self._check_other(spec, OutputType.SCALAR)
        params_dict = {"target": self, "spec": spec}
        return OperationNode(
            self.sds_context,
            "transformencode",
            named_input_nodes=params_dict,
            output_type=OutputType.LIST,
            number_of_outputs=2,
            output_types=[OutputType.MATRIX, OutputType.FRAME],
        )

    def transform_apply(self, spec: "OperationNode", meta: "OperationNode"):
        self._check_frame_op()
        self._check_other(spec, OutputType.SCALAR)
        self._check_other(meta, OutputType.FRAME)
        params_dict = {"target": self, "spec": spec, "meta": meta}
        return OperationNode(
            self.sds_context,
            "transformapply",
            named_input_nodes=params_dict,
            output_type=OutputType.MATRIX,
            number_of_outputs=1,
        )
