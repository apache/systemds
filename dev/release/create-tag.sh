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

################################################################################
##  File:  create-tag.sh
##  Desc:  Release preparation with Maven Release Plugin
################################################################################

# https://stackoverflow.com/q/59895
SELF=$(cd $(dirname $0) && pwd)
. "$SELF/release-utils.sh"

exit_with_usage() {
  local NAME=$(basename $0)
  cat << EOF
usage: $NAME

Tags a SystemDS release on a particular branch.

Inputs are specified with the following environment variables:

ASF_USERNAME - Apache Username
ASF_PASSWORD - Apache Password
GIT_NAME - Name to use with git
GIT_EMAIL - E-mail address to use with git
GIT_BRANCH - Git branch from which to make release
RELEASE_VERSION - Version used in pom files for release
RELEASE_TAG - Name of release tag
NEXT_VERSION - Development version after release
EOF

  exit 1
}

set -e
set -o pipefail

if [[ $@ == *"help"* ]]; then
  exit_with_usage
fi

# docs related to stty 
# https://www.ibm.com/docs/en/aix/7.2?topic=s-stty-command
if [[ -z "$ASF_PASSWORD" ]]; then
  echo 'The environment variable ASF_PASSWORD is not set. Enter the password.'
  echo
  stty -echo && printf "ASF password: " && read ASF_PASSWORD && printf '\n' && stty echo
fi

for env in ASF_USERNAME ASF_PASSWORD RELEASE_VERSION RELEASE_TAG NEXT_VERSION GIT_EMAIL GIT_NAME GIT_BRANCH; do
  if [ -z "${!env}" ]; then
    echo "$env must be set to run this script"
    exit 1
  fi
done

uriencode() { jq -nSRr --arg v "$1" '$v|@uri'; }

declare -r ENCODED_ASF_PASSWORD=$(uriencode "$ASF_PASSWORD")

# git configuration
git config user.name "$GIT_NAME"
git config user.email "$GIT_EMAIL"

printf "$RELEASE_TAG \n"
printf "$RELEASE_VERSION\n"
printf "$NEXT_VERSION"

# options available at https://maven.apache.org/plugins/maven-gpg-plugin/sign-mojo.html
GPG_OPTS="-Dgpg.homedir=$GNUPGHOME -Dgpg.keyname=$GPG_KEY -Dgpg.passphrase=$GPG_PASSPHRASE"

printf "\n -Dgpg.homedir=$GNUPGHOME -Dgpg.keyname=$GPG_KEY \n"


# NOTE:
# 
# When using gpg-plugin in conjunction with release plugin use 
# 
# $ mvn release:perform -Darguments=-Dgpg.passphrase=thephrase
#
# since the system properties of the current maven session are
# not propagated to the forked session automatically.
# 

if is_dry_run; then
  dry_run=true
fi


CMD="mvn --batch-mode -DdryRun=${dry_run} -Dtag=$RELEASE_TAG \
                 -Dresume=false \
                 -DreleaseVersion=$RELEASE_VERSION \
                 -DdevelopmentVersion=$NEXT_VERSION \
                 -Dgpg.keyname=${GPG_KEY} -Dgpg.passphrase=${GPG_PASSPHRASE} \
                 -Darguments=${GPG_OPTS} \
                 release:prepare"

printf "\n #### Executing command: #### \n"
printf "\n $(bold $(greencolor $CMD)) \n\n"

$CMD
