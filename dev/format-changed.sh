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
# Apply the Eclipse formatter (dev/CodeStyle_eclipse.xml) to ONLY the Java files
# you changed -- the same scope the "Java Format" CI check enforces.
#
# A bare `mvn formatter:format` would reformat EVERY .java file under the source
# roots, and the existing tree is not yet fully formatter-clean, so that would
# produce a huge unrelated diff. This script instead formats just the files that
# differ from the base branch (plus your staged/unstaged/untracked changes).
#
# "Changed" = files that differ from the base branch (the merge target), plus
# your staged/unstaged/untracked edits. The base defaults to upstream/main (the
# apache/systemds PR target); run `git fetch upstream main` first so the diff is
# accurate. If the base ref is stale/behind your branch point, the merge-base
# drifts and unrelated files get pulled in -- pass an explicit, current base ref
# in that case.
#
# Usage:
#   dev/format-changed.sh [base-ref]
#     base-ref  branch/commit to diff against (default: upstream/main, else
#               origin/main, else main)
#
set -euo pipefail

FMT_VERSION="2.24.1"
CONFIG="dev/CodeStyle_eclipse.xml"

cd "$(git rev-parse --show-toplevel)"

BASE_REF="${1:-}"
if [ -z "$BASE_REF" ]; then
  for r in upstream/main origin/main main; do
    if git rev-parse --verify --quiet "$r" >/dev/null; then BASE_REF="$r"; break; fi
  done
fi
if [ -z "$BASE_REF" ]; then
  echo "Could not determine a base ref; pass one explicitly: dev/format-changed.sh <base-ref>" >&2
  exit 2
fi

MERGE_BASE="$(git merge-base "$BASE_REF" HEAD 2>/dev/null || echo "$BASE_REF")"

# committed-on-branch + working-tree (staged/unstaged) + untracked new files,
# limited to still-present .java files under the two maven source roots
FILES=$( { git diff --name-only --diff-filter=ACMR "$MERGE_BASE"...HEAD;
           git diff --name-only --diff-filter=ACMR HEAD;
           git ls-files --others --exclude-standard; } \
  | sort -u | grep -E '^src/(main|test)/java/.+\.java$' || true )

if [ -z "$FILES" ]; then
  echo "No changed Java source files to format (base: $BASE_REF)."
  exit 0
fi

INCLUDES=$(echo "$FILES" | sed -E 's#^src/(main|test)/java/##' | paste -sd, -)

echo "Formatting changed Java files (base: $BASE_REF):"
echo "$FILES" | sed 's/^/  /'

mvn -ntp -B \
  net.revelc.code.formatter:formatter-maven-plugin:${FMT_VERSION}:format \
  -Dconfigfile="$CONFIG" \
  -Dmaven.compiler.source=17 -Dmaven.compiler.target=17 \
  -Dformatter.includes="$INCLUDES"
