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

set -euo pipefail

# Usage (from src/main/cpp/jni):
#   chmod +x build_cujava_libs.sh
#   ./build_cujava_libs.sh            # default build dir: ./build, type: Release

BUILD_DIR="${1:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

echo "==> Configuring (BUILD_DIR=$BUILD_DIR, BUILD_TYPE=$BUILD_TYPE)"
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

echo "==> Building"
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j

echo "==> Done. Artifacts should be in ../../lib"
ls -l ../lib/libcujava_runtime.so || true
