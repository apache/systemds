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
    return (
        "You are a JSON extraction assistant. Extract information from the text below.\n"
        "Output ONLY a valid JSON object. Do NOT write code. Do NOT explain.\n"
        "Start your response with { and end with }.\n\n"
        f"Text: {sample.text}\n\n"
        f"Extract these fields: {sample.schema}\n\n"
        "JSON output:"
    )
