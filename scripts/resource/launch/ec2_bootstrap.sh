#!/bin/bash
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

# setup the environment
sudo yum update
sudo yum install -y git
sudo yum install -y java-11-openjdk-devel
sudo yum install -y maven
echo "export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))" | sudo tee -a /etc/profile
source /etc/profile

echo "JDK 11 and MAVEN ware installed successfully"

# install SystemDS
# TODO: point to main repo install the end: git clone https://github.com/apache/systemds.git
git clone https://github.com/lachezar-n/systemds.git
cd systemds
sudo mvn install -Dmaven.test.skip=true
echo "SystemDS installed and setup (single-node) globally"
# install ssm and cloudwatch agents by default (to allow collecting metrics)
sudo yum install amazon-ssm-agent -y
sudo yum install amazon-cloudwatch-agent -y
echo "Cloudwatch agent installed successfully"
# allow checking for installation finished
touch /tmp/systemds_installation_completed


