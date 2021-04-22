---
layout: global
title: SystemML Release Creation Process
description: Description of the SystemML release build process.
displayTitle: SystemML Release Creation Process
---
<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% endcomment %}
-->

* This will become a table of contents (this text will be scraped).
{:toc}

#### Release Creation Guidelines Documentation
Prerequisite: [Project release guidelines](https://github.com/SparkTC/development-guidelines/blob/master/project-release-guidelines.md)



Tips to prepare and release the build

Step 0: Minimum changes and verification to be done before release build process starts.

	1. ReadMe update and “Must Fix” changes are already in.
	2. Performance Test is passing for dataset size of 80GB and below.

Step 1: Prepare the release.

	# Extract latest code to a directory
	<GitRepoHome>

	# Go to dev/release directory
	cd <GitRepoHome>/dev/release

  1.a. Dry Run (this is trial build, will not commit anything in repository).

	e.g. (On Master branch with release candidate rc1, release version 0.15.0, and next development version 1.0.0-SNAPSHOT)
	./release-build.sh --release-prepare --releaseVersion="0.15.0" --developmentVersion="1.0.0-SNAPSHOT" --releaseRc="rc1" --tag="v0.15.0-rc1" --dryRun

	e.g. (On branch-0.15 branch with release candidate rc2, release version 0.15.0, and next development version 0.15.1-SNAPSHOT)
	./release-build.sh --release-prepare --releaseVersion="0.15.0" --developmentVersion="0.15.1-SNAPSHOT" --releaseRc="rc2" --tag="v0.15.0-rc2"  --gitCommitHash="branch-0.15" --dryRun


  1.b. Compile release verification code.

	./release-verify.sh --compile

  1.c. Run license verification.

	./release-verify.sh --verifyLic

  1.d. Run command to do release prepare step (this will commit changes to the repository).  
	This is same as step 1.a, without —dryRun option.

	e.g. (On the Master branch)<br>
	./release-build.sh --release-prepare --releaseVersion="0.15.0" --developmentVersion="1.0.0-SNAPSHOT" --releaseRc="rc1" --tag="v0.15.0-rc1"

	e.g. (On the branch-0.15 branch)
	./release-build.sh --release-prepare --releaseVersion="0.15.0" --developmentVersion="0.15.1-SNAPSHOT" --releaseRc="rc2" --tag="v0.15.0-rc2"  --gitCommitHash="branch-0.15"

  1.e. Verify the release.<br>
	This will verify release on Mac Operating System (OS), assuming these steps are run on Mac OS. It will verify licenses, notice and all other required verification only on Mac OS.
	Verification of licenses and notice is required only on one platform.

	./release-verify.sh --verifyAll


Step 2: Publish the release.

	e.g.
	./release-build.sh --release-publish --gitTag="v0.15.0-rc1"


Step 3: Close the release candidate build on Nexus site.

Visit [NexusRepository](https://repository.apache.org/#stagingRepositories) site.

	Find out SystemML under (Staging Repositories) link. It should be in Open State (status). Close it (button on top left to middle) with proper comment. Once it completes copying, URL will be updated with maven location to be sent in mail.

Step 4: Send mail for voting (dev PMC dev@systemml.apache.org).

Please check [Project release guidelines](https://github.com/SparkTC/development-guidelines/blob/master/project-release-guidelines.md)
or previous mail thread for format/content of the mail.

Step 5: Create a branch based on release to be released.

	# Create a branch based on TAG
	Syntax: git branch <branch name> <Tag Name>
	e.g.    git branch branch-0.15 v0.15.0-rc1

	# Push a branch to master repository
	Syntax: git push origin <branch name>		
	(origin is https://git-wip-us.apache.org/repos/asf/systemml.git)
	e.g.    git push origin branch-0.15


Step 6: If there is failure to get votes then address issues and repeat from step 1.

Step 7: If release has been approved, then make it available for general use for everyone.

	7.a. Move distribution from dev to release (run following commands from command line).

	RELEASE_STAGING_LOCATION="https://dist.apache.org/repos/dist/dev/systemml/"
	RELEASE_STAGING_LOCATION2="https://dist.apache.org/repos/dist/release/systemml/"

	e.g. for SystemML 0.15 rc2 build
	svn move -m "Move SystemML 0.15 from dev to release" $RELEASE_STAGING_LOCATION/0.15.0-rc2  $RELEASE_STAGING_LOCATION2/0.15.0


	7.b. Move Nexus data from dev to release.
	Visit following site and identify release sent for voting in step 3 above. It would be in “closed” state (status).

	https://repository.apache.org/#stagingRepositories

	Click on “Release” button on top middle of the screen and complete the process.

	Note: Release candidates which were not approved can be dropped by clicking “drop” button from top middle of the screen.

	7.c. Update pypi from following site (request someone who has the access).
	https://pypi.python.org/pypi/systemml/

	7.d. Update documents and release notes.

	7.e. Send ANNOUNCE NOTE.
	To:  dev@systemml.apache.org  announce@apache.org
	Subject e.g.
	[ANNOUNCE] Apache SystemML 0.15.0 released.
