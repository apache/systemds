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

source parameters.sh

if [[ ! -d "python_venv" ]]; then
    echo "Creating Python Virtual Enviroment on $HOSTNAME"
    python3 -m venv python_venv
    source "python_venv/bin/activate"
    cd $SYSTEMDS_ROOT
    git pull >/dev/null 2>&1
    mvn clean package -P distribution >/dev/null 2>&1
    cd src/main/python
    pip install wheel >/dev/null 2>&1
    python create_python_dist.py >/dev/null 2>&1
    pip install . | grep "Successfully installed" &&
        echo "Installed Python Systemds Locally" || echo "Failed Installing Python Locally"
fi

## Install remotes
for index in ${!address[*]}; do
    echo "Installing for: ${address[$index]}"
    if [ "${address[$index]}" != "localhost" ]; then
        # Install SystemDS on system.
        ssh -T ${address[$index]} "
        mkdir -p github;
        cd github;
        if [[ ! -d 'systemds' ]]; then  git clone https://github.com/apache/systemds.git  > /dev/null 2>&1; fi;
        cd systemds;
        git reset --hard origin/master > /dev/null 2>&1;
        git pull > /dev/null 2>&1; 
        mvn clean package  -P distribution > /dev/null 2>&1;
        echo 'Installed Systemds on' \$HOSTNAME;
        cd \$HOME
        mkdir -p ${remoteDir}
        " &
    fi
done

wait
