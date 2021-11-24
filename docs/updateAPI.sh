#/bin/bash
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

# stop the script at first error
set -e

curFolder=${PWD##*/}

if [ $curFolder != "docs" ]; then
    echo "must be run from docs folder"
    exit
else
    echo "creating API docs"
    rm -r api/java
    ## JAVA Docs
    cd ..
    mvn clean javadoc:javadoc -P distribution
    mkdir -p docs/api/java
    cp -r target/site/apidocs/* docs/api/java
    cd docs

    ## Python Docs
    rm -r api/python
    mkdir api/python
    cd ../src/main/python/docs
    rm -f -r build
    make html
    cd ../../../../
    cp -r src/main/python/docs/build/html/* docs/api/python
    ## Hack The folder names, becuse Jekyll ignores folders starting with underscore.
    cd docs/api/python
    mv _static static
    find . -type f -exec sed -i 's/_static/static/g' {} +
    mv _sources sources
    find . -type f -exec sed -i 's/_sources/sources/g' {} +
    
    if [[ -d "_images" ]]
    then
      mv _images images
      find . -type f -exec sed -i 's/_images/images/g' {} +
    fi
    
    cd ../../
fi

exit 0

