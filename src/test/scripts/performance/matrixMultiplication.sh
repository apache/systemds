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


mvn package 2>&1 > /dev/null

cm="java --add-modules=jdk.incubator.vector -jar -XX:+UseNUMA target/systemds-3.4.0-SNAPSHOT-perf.jar 1009"

$cm 5 5 5 1 1 true
$cm 500 5 5 1 1 true
$cm 5 500 5 1 1 true
$cm 5 5 500 1 1 true

$cm 100 100 100 1 1 true
$cm 1000 100 100 1 1 true
$cm 100 1000 100 1 1 true
$cm 100 100 1000 1 1 true

$cm 1000 1000 1000 1 1 true

$cm 10000 1000 1000 1 1 true
$cm 1000 10000 1000 1 1 true
$cm 1000 1000 10000 1 1 true

$cm 10000 10000 10000 1 1 false