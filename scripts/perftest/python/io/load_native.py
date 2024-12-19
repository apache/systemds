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
    ]
)


run = "\n".join(
    [
        "node = ctx.read(src)",
        "script = DMLScript(ctx)",
        "script.build_code(node)",
        "script.execute()",
    ]
)


def main(args):
    gvars = {"src": args.src, "ctx": SystemDSContext(logging_level=10, py4j_logging_level=50)}
    print(timeit.timeit(run, setup, globals=gvars, number=args.number))
    gvars["ctx"].close()


if __name__ == "__main__":
    description = "Benchmarks time spent loading data into systemds"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("src")
    parser.add_argument("number", type=int, help="number of times to load the data")
    args = parser.parse_args()
    main(args)
