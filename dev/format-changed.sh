#!/usr/bin/env bash
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
#
# Apply the Eclipse formatter (dev/CodeStyle_eclipse.xml) to ONLY the lines you
# edited -- the exact scope the "Java Format" CI check enforces.
#
# A bare `mvn formatter:format` would reformat EVERY .java file under the source
# roots, and even scoping to changed *files* would reformat their pre-existing
# (not-yet-clean) lines. The existing tree is not fully formatter-clean, so both
# would produce a large unrelated diff. This delegates to dev/format_changed.py,
# which formats each changed file but keeps only the changes that land on the
# lines you actually edited.
#
# "Changed" = lines that differ from the base branch (the merge target),
# including your committed-on-branch, staged, unstaged and untracked edits. The
# base defaults to upstream/main (the apache/systemds PR target); run
# `git fetch upstream main` first so the diff is accurate, or pass an explicit,
# current base ref if the default is stale/behind your branch point.
#
# Usage:
#   dev/format-changed.sh [base-ref]
#     base-ref  branch/commit to diff against (default: upstream/main, else
#               origin/main, else main)
#
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

exec python3 dev/format_changed.py --fix "$@"
