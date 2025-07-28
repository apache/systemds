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

import numpy as np
import torch


def numpy_dtype_to_torch_dtype(dtype):
    """
    Convert a NumPy dtype (or dtype string) to the corresponding PyTorch dtype.
    Raises ValueError if the dtype is not supported.
    """
    if isinstance(dtype, torch.dtype):
        return dtype

    mapping = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.float16: torch.bfloat16,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
    }

    np_dtype = np.dtype(dtype)
    if np_dtype.type in mapping:
        return mapping[np_dtype.type]
    else:
        raise ValueError(f"No corresponding torch dtype for NumPy dtype {np_dtype}")
