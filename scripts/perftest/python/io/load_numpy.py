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
import timeit
from systemds.context import SystemDSContext

setup = "\n".join(
    [
        "from systemds.script_building.script import DMLScript",
        "import numpy as np",
        "import os",
        "if os.path.isdir(src):",
        "    files = [os.path.join(src, f) for f in os.listdir(src)]",
        "    array = np.concatenate([np.loadtxt(f, delimiter=',') for f in files])",
        "else:",
        "    array = np.loadtxt(src, delimiter=',')",
        "if dtype is not None:",
        "    array = array.astype(dtype)",
    ]
)


run = "\n".join(
    [
        "matrix_from_np = ctx.from_numpy(array)",
        "script = DMLScript(ctx)",
        "script.add_input_from_python('test', matrix_from_np)",
        "script.execute()",
    ]
)


dtype_choices = [
    "double",
    "float",
    "long",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "string",
    "bool",
]


def main(args):
    gvars = {"src": args.src, "dtype": args.dtype, "ctx": SystemDSContext(logging_level=10, py4j_logging_level=50)}
    print(timeit.timeit(run, setup, globals=gvars, number=args.number))
    gvars["ctx"].close()


if __name__ == "__main__":
    description = "Benchmarks time spent loading data into systemds"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("src")
    parser.add_argument("number", type=int, help="number of times to load the data")
    help_force_dtype = (
        "optionally cast all columns to one of the dtype choices in numpy"
    )
    parser.add_argument(
        "--dtype",
        choices=dtype_choices,
        required=False,
        default=None,
        help=help_force_dtype,
    )
    args = parser.parse_args()

    if args.dtype == "string":  # numpy has no "string" dtype, convert to "str"
        args.dtype = "str"

    main(args)
