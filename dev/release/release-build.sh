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

function exit_with_usage {
  cat << EOF

release-build - Creates build distributions from a git commit hash or from HEAD.

SYNOPSIS

usage: release-build.sh [--release-prepare | --release-publish | --release-snapshot]

DESCRIPTION

Use maven infrastructure to create a project release package and publish
to staging release location (https://dist.apache.org/repos/dist/dev/incubator/systemml/)
and maven staging release repository.

--release-prepare --releaseVersion="0.11.0-incubating" --developmentVersion="0.11.0-SNAPSHOT" [--releaseRc="rc1"] [--tag="v0.11.0-incubating"] [--gitCommitHash="a874b73"]
This form execute maven release:prepare and upload the release candidate distribution
to the staging release location.

--release-publish --gitCommitHash="a874b73"
Publish the maven artifacts of a release to the Apache staging maven repository.

--release-snapshot [--gitCommitHash="a874b73"]
Publish the maven snapshot artifacts to Apache snapshots maven repository

OPTIONS

--releaseVersion     - Release identifier used when publishing
--developmentVersion - Release identifier used for next development cyce
--releaseRc          - Release RC identifier used when publishing, default 'rc1'
--tag                - Release Tag identifier used when taging the release, default 'v$releaseVersion'
--gitCommitHash      - Release tag or commit to build from, default master HEAD
--dryRun             - Dry run only, mostly used for testing.

A GPG passphrase is expected as an environment variable

GPG_PASSPHRASE - Passphrase for GPG key used to sign release

EXAMPLES

release-build.sh --release-prepare --releaseVersion="0.11.0-incubating" --developmentVersion="0.12.0-SNAPSHOT"
release-build.sh --release-prepare --releaseVersion="0.11.0-incubating" --developmentVersion="0.12.0-SNAPSHOT" --releaseRc="rc1" --tag="v0.11.0-incubating-rc1"
release-build.sh --release-prepare --releaseVersion="0.11.0-incubating" --developmentVersion="0.12.0-SNAPSHOT" --releaseRc="rc1" --tag="v0.11.0-incubating-rc1"  --gitCommitHash="a874b73" --dryRun

release-build.sh --release-publish --gitCommitHash="a874b73"
release-build.sh --release-publish --gitTag="v0.11.0-incubating-rc1"

release-build.sh --release-snapshot
release-build.sh --release-snapshot --gitCommitHash="a874b73"

EOF
  exit 1
}

set -e

if [ $# -eq 0 ]; then
  exit_with_usage
fi


# Process each provided argument configuration
while [ "${1+defined}" ]; do
  IFS="=" read -ra PARTS <<< "$1"
  case "${PARTS[0]}" in
    --release-prepare)
      GOAL="release-prepare"
      RELEASE_PREPARE=true
      shift
      ;;
    --release-publish)
      GOAL="release-publish"
      RELEASE_PUBLISH=true
      shift
      ;;
    --release-snapshot)
      GOAL="release-snapshot"
      RELEASE_SNAPSHOT=true
      shift
      ;;
    --gitCommitHash)
      GIT_REF="${PARTS[1]}"
      shift
      ;;
    --gitTag)
      GIT_TAG="${PARTS[1]}"
      shift
      ;;
    --releaseVersion)
      RELEASE_VERSION="${PARTS[1]}"
      shift
      ;;
    --developmentVersion)
      DEVELOPMENT_VERSION="${PARTS[1]}"
      shift
      ;;
    --releaseRc)
      RELEASE_RC="${PARTS[1]}"
      shift
      ;;
    --tag)
      RELEASE_TAG="${PARTS[1]}"
      shift
      ;;
    --dryRun)
      DRY_RUN="-DdryRun=true"
      shift
      ;;

    *help* | -h)
      exit_with_usage
     exit 0
     ;;
    -*)
     echo "Error: Unknown option: $1" >&2
     exit 1
     ;;
    *)  # No more options
     break
     ;;
  esac
done


if [[ -z "$GPG_PASSPHRASE" ]]; then
    echo 'The environment variable GPG_PASSPHRASE is not set. Enter the passphrase to'
    echo 'unlock the GPG signing key that will be used to sign the release!'
    echo
    stty -echo && printf "GPG passphrase: " && read GPG_PASSPHRASE && printf '\n' && stty echo
fi

if [[ "$RELEASE_PREPARE" == "true" && -z "$RELEASE_VERSION" ]]; then
    echo "ERROR: --releaseVersion must be passed as an argument to run this script"
    exit_with_usage
fi

if [[ "$RELEASE_PREPARE" == "true" && -z "$DEVELOPMENT_VERSION" ]]; then
    echo "ERROR: --developmentVersion must be passed as an argument to run this script"
    exit_with_usage
fi

if [[ "$RELEASE_PUBLISH" == "true"  ]]; then
    if [[ "$GIT_REF" && "$GIT_TAG" ]]; then
        echo "ERROR: Only one argumented permitted when publishing : --gitCommitHash or --gitTag"
        exit_with_usage
    fi
    if [[ -z "$GIT_REF" && -z "$GIT_TAG" ]]; then
        echo "ERROR: --gitCommitHash OR --gitTag must be passed as an argument to run this script"
        exit_with_usage
    fi
fi

if [[ "$RELEASE_PUBLISH" == "true" && "$DRY_RUN" ]]; then
    echo "ERROR: --dryRun not supported for --release-publish"
    exit_with_usage
fi

if [[ "$RELEASE_SNAPSHOT" == "true" && "$DRY_RUN" ]]; then
    echo "ERROR: --dryRun not supported for --release-publish"
    exit_with_usage
fi

# Commit ref to checkout when building
GIT_REF=${GIT_REF:-master}
if [[ "$RELEASE_PUBLISH" == "true" && "$GIT_TAG" ]]; then
    GIT_REF="tags/$GIT_TAG"
fi

BASE_DIR=$(pwd)
RELEASE_WORK_DIR=$BASE_DIR/target/release

MVN="mvn"
PUBLISH_PROFILES="-Pdistribution,rat"

if [ -z "$RELEASE_RC" ]; then
  RELEASE_RC="rc1"
fi

if [ -z "$RELEASE_TAG" ]; then
  RELEASE_TAG="v$RELEASE_VERSION-$RELEASE_RC"
fi

RELEASE_STAGING_LOCATION="https://dist.apache.org/repos/dist/dev/incubator/systemml/"


echo "  "
echo "-------------------------------------------------------------"
echo "------- Release preparation with the following parameters ---"
echo "-------------------------------------------------------------"
echo "Executing           ==> $GOAL"
echo "Git reference       ==> $GIT_REF"
echo "release version     ==> $RELEASE_VERSION"
echo "development version ==> $DEVELOPMENT_VERSION"
echo "rc                  ==> $RELEASE_RC"
echo "tag                 ==> $RELEASE_TAG"
if [ "$DRY_RUN" ]; then
   echo "dry run ?           ==> true"
fi
echo "  "
echo "Deploying to :"
echo $RELEASE_STAGING_LOCATION
echo "  "

function checkout_code {
    # Checkout code
    rm -rf $RELEASE_WORK_DIR
    mkdir $RELEASE_WORK_DIR
    cd $RELEASE_WORK_DIR
    git clone https://git-wip-us.apache.org/repos/asf/incubator-systemml.git
    cd incubator-systemml
    git checkout $GIT_REF
    git_hash=`git rev-parse --short HEAD`
    echo "Checked out SystemML git hash $git_hash"

    git clean -d -f -x
    #rm .gitignore
    #rm -rf .git

    cd "$BASE_DIR" #return to base dir
}

if [[ "$RELEASE_PREPARE" == "true" ]]; then
    echo "Preparing release $RELEASE_VERSION"
    # Checkout code
    checkout_code
    cd $RELEASE_WORK_DIR/incubator-systemml

    # Build and prepare the release
    $MVN $PUBLISH_PROFILES release:clean release:prepare $DRY_RUN -Darguments="-Dgpg.passphrase=\"$GPG_PASSPHRASE\" -DskipTests" -DreleaseVersion="$RELEASE_VERSION" -DdevelopmentVersion="$DEVELOPMENT_VERSION" -Dtag="$RELEASE_TAG"

    cd $RELEASE_WORK_DIR

    if [ -z "$DRY_RUN" ]; then
        svn co $RELEASE_STAGING_LOCATION svn-release-staging
        mkdir -p svn-release-staging/$RELEASE_VERSION-$RELEASE_RC
        cp $RELEASE_WORK_DIR/incubator-systemml/target/systemml-* svn-release-staging/$RELEASE_VERSION-$RELEASE_RC/

        cd svn-release-staging/$RELEASE_VERSION-$RELEASE_RC/
        rm -f *.asc
        for i in *.jar *.zip *.gz *.tgz; do gpg --output $i.asc --detach-sig --armor $i; done
        rm -f *.md5
        for i in *.jar *.zip *.gz *.tgz; do openssl md5 -hex $i | sed 's/MD5(\([^)]*\))= \([0-9a-f]*\)/\2 *\1/' > $i.md5; done
        rm -f *.sha
        for i in *.jar *.zip *.gz *.tgz; do shasum $i > $i.sha; done

        cd .. #exit $RELEASE_VERSION-$RELEASE_RC/

        svn add $RELEASE_VERSION-$RELEASE_RC/
        svn ci -m"Apache SystemML $RELEASE_VERSION-$RELEASE_RC"
    fi


    cd "$BASE_DIR" #exit target

    exit 0
fi


if [[ "$RELEASE_PUBLISH" == "true" ]]; then
    echo "Preparing release $RELEASE_VERSION"
    # Checkout code
    checkout_code
    cd $RELEASE_WORK_DIR/incubator-systemml

    #Deploy scala 2.10
    mvn -DaltDeploymentRepository=apache.releases.https::default::https://repository.apache.org/service/local/staging/deploy/maven2 clean package gpg:sign install:install deploy:deploy -DskiptTests -Darguments="-DskipTests" -Dgpg.passphrase=$GPG_PASSPHRASE $PUBLISH_PROFILES

    cd "$BASE_DIR" #exit target

    exit 0
fi


if [[ "$RELEASE_SNAPSHOT" == "true" ]]; then
    # Checkout code
    checkout_code
    cd $RELEASE_WORK_DIR/incubator-systemml

    CURRENT_VERSION=$($MVN help:evaluate -Dexpression=project.version \
    | grep -v INFO | grep -v WARNING | grep -v Download)

    # Publish Bahir Snapshots to Maven snapshot repo
    echo "Deploying SystemML SNAPSHOT at '$GIT_REF' ($git_hash)"
    echo "Publish version is $CURRENT_VERSION"
    if [[ ! $CURRENT_VERSION == *"SNAPSHOT"* ]]; then
        echo "ERROR: Snapshots must have a version containing SNAPSHOT"
        echo "ERROR: You gave version '$CURRENT_VERSION'"
        exit 1
    fi

    #Deploy scala 2.10
    $MVN -DaltDeploymentRepository=apache.snapshots.https::default::https://repository.apache.org/content/repositories/snapshots clean package gpg:sign install:install deploy:deploy -DskiptTests -Darguments="-DskipTests" -Dgpg.passphrase=$GPG_PASSPHRASE $PUBLISH_PROFILES

    cd "$BASE_DIR" #exit target
    exit 0
fi


cd "$BASE_DIR" #return to base dir
echo "ERROR: wrong execution goals"
exit_with_usage
