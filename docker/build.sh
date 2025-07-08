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

# Build the docker containers

# The first build is for running systemds through docker.
# docker image build -f docker/sysds.Dockerfile -t apache/systemds:latest .

# The second build is for testing systemds. This image installs the R dependencies needed to run the tests.
# docker image build -f docker/testsysds.Dockerfile -t apache/systemds:testing-latest .

# The third build is python docker for systemds. 
# docker image build -f docker/pythonsysds.Dockerfile -t apache/systemds:python-nightly .

# You might want to prune the docker system afterwards using
# docker system prune
