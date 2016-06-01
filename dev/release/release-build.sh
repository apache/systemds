#!/usr/bin/env bash

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


# Display commands as they are executed
set -x

# BUILD and install in the current directory
ROOT=`pwd`
BUILD=`pwd`/target
RELEASE=`pwd`/target/RELEASE

#For testing
#mvn clean verify gpg:sign install:install deploy:deploy -DaltDeploymentRepository=id::default::file:$RELEASE -Pdistribution,rat  -Dgpg.skip -DskipTests -Darguments="-DskipTests"

#For publishing
#mvn clean verify gpg:sign install:install deploy:deploy -Dgpg.passphrase=XXX -Pdistribution,rat -DskipTests -Darguments="-DskipTests"

mvn release:prepare -Pdistribution,rat -DskipTests -Darguments="-DskipTests" -DreleaseVersion="0.10.0-incubating" -DdevelopmentVersion="0.11.0-incubating-SNAPSHOT" -Dtag="0.10.0-incubating"

mvn release:perform gpg:sign install:install deploy:deploy -Dgpg.passphrase=XXX -Pdistribution,rat -DskipTests -Darguments="-DskipTests" -DreleaseVersion="0.10.0-incubating" -DdevelopmentVersion="0.11.0-incubating-SNAPSHOT" -Dtag="0.10.0-incubating"


mkdir $RELEASE/
cp $BUILD/systemml-* $RELEASE/
cd $RELEASE

# sign
#for i in *.zip *.gz; do gpg --output $i.asc --detach-sig --armor $i; done
for i in *.zip *.gz; do openssl md5 -hex $i | sed 's/MD5(\([^)]*\))= \([0-9a-f]*\)/\2 *\1/' > $i.md5; done
for i in *.jar; do openssl md5 -hex $i | sed 's/MD5(\([^)]*\))= \([0-9a-f]*\)/\2 *\1/' > $i.md5; done

cp $BUILD/rat.txt $RELEASE/

# copy to apache for review
# scp $RELEASE/* lresende@people.apache.org:/home/lresende/public_html/systemml/0.9.0