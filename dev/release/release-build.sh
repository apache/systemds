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
##  File:  release-build.sh
##  Desc:  Create Release artifacts
################################################################################

SELF=$(cd $(dirname $0) && pwd)
. "$SELF/release-utils.sh"

exit_with_usage() {

  cat << EOF
usage: release-build.sh <package|docs>

Create build deliverables from a commit
Top level targets are
  - package: create binary packages and commit them
             to staging repo.
  - docs: Build docs and commit them to staging repo
  - publish-release: Build maven artifacts and publish to Maven Release Repo
  - publish-staging: Publish to staging repository

GIT_REF - Release tag or commit to build from
PACKAGE_VERSION - Release identifier in top level package directory (eg. 2.1.2-rc1)
BUILD_VERSION - (optional) Version being built (eg. 2.1.2)
ASF_USERNAME - Username of ASF committer
ASF_PASSWORD - Password of ASF committer account
GPG_KEY - GPG key used to sign release artifacts
GPG_PASSPHRASE - Passphrase for GPG key
EOF
  exit 1
}

if [ $# -eq 0 ]; then
  echo "usage: release-build.sh <docs|publish-release>"
fi

error() {
  echo "$*"
  exit 1
}

if [ $# -eq 0 ]; then
  exit_with_usage
fi


if [[ $@ == *"help"* ]]; then
  exit_with_usage
fi


# Build docs (production)
if [[ "$1" == "docs" ]]; then
  cd systemds
  echo "Building SystemDS docs"

  cd docs

  bundle install
  PRODUCTION=1 RELEASE_VERSION="$RELEASE_VERSION" bundle exec jekyll build
fi

GPG_OPTS="-Dgpg.keyname=${GPG_KEY} -Dgpg.passphrase=${GPG_PASSPHRASE}"

cat <<EOF >../tmp-settings.xml
<settings><servers><server>
<id>apache.snapshots.https</id><username>${ASF_USERNAME}</username>
<password>${ASF_PASSWORD}</password>
</server>
<server>
<id>apache.releases.https</id><username>${ASF_USERNAME}</username>
<password>${ASF_PASSWORD}</password>
</server>
</servers>
</settings>
EOF


# Publishing to Sonatype repo, details:
NEXUS_ROOT=https://repository.apache.org/service/local/staging
NEXUS_PROFILE=1486a6e8f50cdf

# Apache SVN Repo, details:
RELEASE_STAGING_LOCATION="https://dist.apache.org/repos/dist/dev/systemds"
DEST_DIR_NAME="$PACKAGE_VERSION"

# NOTE:
# 1. Build files will be saved to this folder.
# This folder will be used by `publish-release`
# 
# 2. this directory is passed via `file` protocol with
#  file:///${path} (3 slashes, specifies empty name)
#  refer: https://en.wikipedia.org/wiki/File_URI_scheme#How_many_slashes.3F
mkdir temp
tmp_repo=$(mktemp -d temp/systemds-repo-tmp-XXXXX)

if [[ "$1" == "publish-release" ]]; then

  # cd systemds
  
  # Publishing spark to Maven Central Repo
  printf "\nRelease version is ${PACKAGE_VERSION} \n"
  
  mvn versions:set -DnewVersion=${RELEASE_VERSION}

  # if ! is_dry_run; then
    printf "Creating a Nexus staging repository \n"
    promote_request="<promoteRequest><data><description>Apache SystemDS</description></data></promoteRequest>"
    out=$(curl -X POST -d "$promote_request" -u $ASF_USERNAME:$ASF_PASSWORD \
      -H "Content-Type:application/xml" -v \
      $NEXUS_ROOT/profiles/$NEXUS_PROFILE/start)
    staged_repository_id=$(echo $out | sed -e "s/.*\(orgapachesystemds-[0-9]\{4\}\).*/\1/")
  # fi

  cat <<EOF >../tmp-settings-nexus.xml
<settings>
<activeProfiles>
    <activeProfile>local-temp</activeProfile>
  </activeProfiles>

  <profiles>
    <profile>
      <id>local-temp</id>
      <repositories>
        <repository>
          <id>local-temp</id>
          <url>file:///$PWD/${tmp_repo}</url>
        </repository>
      </repositories>
    </profile>
  </profiles>
</settings>
EOF

  mvn --settings ../tmp-settings-nexus.xml -Pdistribution deploy \
    -DaltDeploymentRepository=local-temp::default::file:///$PWD/${tmp_repo} \
    -Daether.checksums.algorithms='SHA-512,SHA-1,MD5'

  pushd "${tmp_repo}/org/apache/systemds"
  

  # if ! is_dry_run; then
    # upload files to nexus repo
    nexus_upload_id=$NEXUS_ROOT/deployByRepositoryId/$staged_repository_id
    printf "\nUpload files to $nexus_upload_id \n"

    for file in $(find . -type f)
    do
      # strip leading ./
      file_short=$(echo $file | sed -e "s/\.\///")
      dest_url="$nexus_upload_id/org/apache/systemds/$file_short"
      printf "\nUploading $file_short \n"
      curl -u $ASF_USERNAME:$ASF_PASSWORD --upload-file $file_short $dest_url
    done

    # Promote the staging repository
    promote_request="<promoteRequest><data><stagedRepositoryId>$staged_repository_id</stagedRepositoryId></data></promoteRequest>"
    out=$(curl -X POST -d "$promote_request" -u $ASF_USERNAME:$ASF_PASSWORD \
      -H "Content-Type:application/xml" -v \
      $NEXUS_ROOT/profiles/$NEXUS_PROFILE/finish)
    printf "Closed Nexus staging repository: $staged_repository_id"

    printf "\nAfter release vote passes make sure to hit release button.\n"

  # else
  #   printf "Files will uploaded to Nexus Repo at this step."
  # fi
    
    printf "\n ============== "
    printf "\n Upload artifacts to dist.apache.org \n"
    
    svn co --depth=empty $RELEASE_STAGING_LOCATION svn-systemds

    if ! is_dry_run; then
      stage_dir="svn-systemds/${PACKAGE_VERSION}"
      mkdir -p $stage_dir
    else
      stage_dir=$(mktemp -d svn-systemds/${DEST_DIR_NAME}-temp-XXXX)
    fi

    printf "\nCopy the release tarballs to svn repo \n"
    ls *
    
    # Remove extra files generated
    # Keep only .zip, .tgz, and javadoc
    find systemds -type f | grep -v -e \.zip -e \.tgz -e javadoc | xargs rm
    eval cp systemds/${RELEASE_VERSION}/systemds-* "${stage_dir}"
    svn add "${stage_dir}"
    
    eval cd svn-systemds
    svn ci --username "$ASF_USERNAME" --password "$ASF_PASSWORD" -m"Apache SystemDS $SYSTEMDS_PACKAGE_VERSION" --no-auth-cache
    eval cd ..
    rm -rf svn-systemds

  popd

  # NOTE: Do not delete any generated release artifacts
  # rm -rf "${tmp_repo}"
  eval cd ..
  exit 0
fi

