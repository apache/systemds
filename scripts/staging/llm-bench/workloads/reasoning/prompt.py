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

from typing import Any, Dict

from .loader import Sample


def make_prompt(sample: Sample, cfg: Dict[str, Any]) -> str:
    """Format prompt based on puzzle type to match the dataset properly."""
    if sample.puzzle_type == "boolean_reasoning":
        # BoolQ: reading comprehension with yes/no answer
        return (
            f"{sample.puzzle}\n\n"
            "Think step-by-step, then state your final answer as "
            "exactly 'Yes' or 'No'."
        )
    if sample.puzzle_type == "logical_reasoning":
        # LogiQA: multiple-choice logical reasoning
        return (
            f"{sample.puzzle}\n\n"
            "Think step-by-step, then state your final answer as "
            "a single letter (A, B, C, or D)."
        )
    # Toy / other: generic reasoning prompt
    return (
        "Solve the following problem step-by-step. "
        "Show your reasoning, then state the final answer.\n\n"
        f"{sample.puzzle}\n"
    )
