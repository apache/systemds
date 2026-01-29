<!-- 
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
-->

# Data Integration and Large-Scale Analysis
## Overview

This README presents the project 3878 and serves as index for the documentation.

The goal of SYSTEMDS-3878 as presented in [jira](https://issues.apache.org/jira/browse/SYSTEMDS-3878) is to "Improve Docker Security":
>This task is to improve the security of the Docker images we provide. Currently we get an 'F' evaluation on DockerHub, and we would like instead to provide secure images.
https://hub.docker.com/repository/docker/apache/systemds/general

<!-- TODO Table of contents -->

## Working on apache/systemds

### Create a fork

1. Fork systemds github on own account
2. Invite project member
1. Create SSH-key locally on computer
1. Add public key to github account
1. Clone project on computer (git clone git@\<link>.git)
1. Get a local version of the systemds project on the computer

### Commiting

Extract from [CONTRIBUTING.md](https://github.com/qschnee/systemds/blob/main/CONTRIBUTING.md).

A commit or PR description is a public record of **what** change is being made and **why**.

#### Structure of the description

##### First Line

1. A summary of what the changeset.
2. A complete sentence, crafted as though it was an order.
    - an imperative sentence
    - Writing the rest of the description as an imperative is optional.
3. Follow by an empty line.

##### Body

It consists of the following.

1. A brief description of the problem solved.
2. Why this is the best approach?.
3. Shortcomings to the approach, if any (important!).

Additional info

4. background information
   - bug/issue/jira numbers
   - benchmark/test results
   - links to design documents
   - code attributions
5. Include enough context for
   - reviewers
   - future readers to understand the Changes.
6. Add PR number, like `Closes #1000`.

#### Example for this project:

```text
[MINOR][SYSTEMDS-3878] Fix <file>.Dockerfile vulnerability (<bug>)

This commit fixes the following vulnerability identified by docker scout cves: <...>. The following changes have been made on the Dockerfile: <...>.
<Sources/justifications>
<Shortcomings>
<new CVE if not fully resolved>
```

## Docker scout cves

We used the tool docker scout cves to identify and solve vulenrabilities in the systemds image build from the Dockerfile.

From the man page:
>The docker scout cves command analyzes a software artifact for vulnerabilities. \
If no image is specified, the most recently built image is used. \
[...] \
The tool analyzes the provided software artifact, and generates a vulnerability report.

### Usage

``` sh
cd path/to/systemds
./docker/build.sh
docker scout cves --details --format markdown -o docker/scout_results/sysds_outputX.md --epss apache/systemds
```

To identify the location of tricky packages, using the json output with `--format sarif` helps by showing the path to the vulnerable package (which is not shown in the "human-readable" markdown output).

### Command details

#### Local analysis

To use docker scout and see if local changes succesfully solved a vulnerability, it is necessary to modify `sysds.Dockerfile`. \
`systemds` is cloned from github instead of using the local systemds project. This change should be reverted and rever merged to the master branch.
```Dockerfile
# Build the system
# RUN git clone --depth 1 https://github.com/apache/systemds.git systemds && \
# 	cd /usr/src/systemds/ && \
# 	mvn --no-transfer-progress clean package -P distribution

# Copy the local SystemDS source into the image
COPY . /usr/src/systemds
# Build SystemDS
RUN cd /usr/src/systemds && \
   mvn --no-transfer-progress clean package -P distribution
```

#### Options

<https://docs.docker.com/reference/cli/docker/scout/cves/>

1. build systemds project \
   `./docker/build.sh`
   - builds the image with the tag: `apache/systemds:latest`
   - default runs: `docker image build -f docker/sysds.Dockerfile -t apache/systemds:latest .`
   - modify `docker/build.sh` to change the selected `Dockerfile` for the build
2. scout \
   `docker scout cves --details --format markdown -o <docker_subdirecory>/scout_results/<file>_output0.md --epss apache/systemds` (from systemds directory)
   - `--details`: verbose
   - `--format markdown`: output in markdown format
   - `-o <path_to_file>`: write output to file
   - `-epss`: show epss score
      - `--epss --epss-score 0.1`: filter for vulnerabilities that have more than 10% probability to be exploited.

#### Troubleshoot
Docker scout does not run because `/tmp` is full:
- `df -h`: to see usage of partitions.
- `du -h /home/schnee/DIA-sysds-scout-tmp/` to see directory usage.
- `docker scout cache df` and `docker scout cache prune` to view and clear scout cache.

Change default tmp partition to a bigger filesystem
- `export TMPDIR=/absolute/path/to/sysds-scout-tmp`
- `export DOCKER_SCOUT_CACHE_DIR=/absolute/path/to/sysds-scout-tmp`

<details>

   <summary> Directly from filesystem (Dockerfile)</summary>

   > Does Not Work

   No vulnerability found

   ```bash
   cd project_root_directory
   docker scout cves --format markdown -o scout_results/output0.md --epss fs://docker_subdirecory/file.Dockerfile
   ```

</details>

### Solve vulnerabilities

By using the CVE code and reading its description, most vulnerabilities have solutions or workarounds.

#### Upgrading the related package

One way to solve a vulerability is simply to upgrade the package the CVE happens in.

> Sometimes, the package that raised the CVE is not included in the `pom.xml`. This can happen if the package is a dependency of another imported package. \
> `mvn dependency:tree` (example output can be found [here](systemds-3878_summary-of-changes.md#output-of-mvn-dependencytree)) shows all imported packages and implicit dependencies imported by other packets.

#### Remove or switch the packet

If the vulnerability comes from a transitive dependency:
1. update the parent dependency to a newer version which doesn't use the vulnerable packet. \
This can be verified in Maven Repository <https://mvnrepository.com/> by searching the parent packet.
1. exclude the vulnerable package from the parent and explicitly import a newer version of the vulnerable package. \
By doing this, the parent dependency will use the explicit version instead of the vulnerable one.

### Inspection commands to find tricky packages

As mentioned before, using the `sarif` format in scout could help identify where the vulnerale package comes from.

If this does not help, it is possible to inspect the image directly to find the related jar. Here, `apache/zookeeper` is used as reference`:
- copy container filesystem to analyze jars: 

  ```sh
  docker create --name temp apache/systemds
  docker export temp > img.tar
  mkdir filesystem_img
  tar -xf img.tar -C filesystem_img
  #Analysis
  rm -rf img.tar temp filesystem_img
  docker rm temp
  ```
  - Find all references to zookeeper

    `find filesystem_img -name '*zookeeper*'`
  - List all libs 
    
    `ls -la ./filesystem_img/systemds/target/lib/`
  - Inspect the jars in the image for references to zookeeper

    `find ./filesystem_img -name '*.jar' -exec sh -c 'jar tf {} | grep -i zookeeper && echo "--- Found in {}"' \;`
- Inspect the given jar for references to zookeeper 

  `jar tf ~/.m2/repository/org/apache/hadoop/hadoop-common/3.3.6/hadoop-common-3.3.6.jar | grep -i zookeeper`

### `docker scout` helloworld example

To test if docker scout works correctly on your machine, you can create a simple "Hello World" Dockerfile.

```bash
mkdir scout_hello_world
cd scout_hello_world
echo -e "FROM ubuntu:latest\nRUN apt-get update && apt-get install -y curl\nCMD ["curl", "https://example.com"]" > Dockerfile
docker build -t scout_hello_world_img .
# optionally check image: docker images | grep hello_world
docker scout cves scout_hello_world_img
```

1. create and enter new directory
2. Create new file `Dockerfile` and write content into it
1. Build docker image from current directory `.` \
   With name scout_hello_wolrd (`-t`)
1. Scout the image with the name

The output can be viewed in the [appendix](#docker-scout-helloworld-output)


## Student Project SYSTEMDS-3878

### Results

We managed to solve a lot of CVEs: 

| FROM: | <img alt="critical: 4" src="https://img.shields.io/badge/critical-4-8b1924"/> | <img alt="high: 29" src="https://img.shields.io/badge/high-29-e25d68"/> | <img alt="medium: 36" src="https://img.shields.io/badge/medium-36-fbb552"/> | <img alt="low: 9" src="https://img.shields.io/badge/low-9-fce1a9"/> | <img alt="unspecified: 1" src="https://img.shields.io/badge/unspecified-1-lightgrey"/>|
| :--- | :--- | :--- | :--- | :--- | :--- |
| TO: | <img alt="critical: 0" src="https://img.shields.io/badge/critical-0-lightgrey"/> | <img alt="high: 6" src="https://img.shields.io/badge/high-6-e25d68"/> | <img alt="medium: 8" src="https://img.shields.io/badge/medium-9-fbb552"/> | <img alt="low: 1" src="https://img.shields.io/badge/low-1-fce1a9"/> | <img alt="unspecified: 1" src="https://img.shields.io/badge/unspecified-1-lightgrey"/> |

The `scout` reports can be viewed in details: [scan-before-fixes](./scan-before-fixes/README.md), [scan-after-fixes](./scan-after-fixes/README.md).

[unfixed-vulnerabilities/README.md](./unfixed-vulnerabilities/README.md) has been written to explain the last vulnerabilities that hove not been fixed and why.

### Testing
Running `mvn clean verify` after our changes returns the same output.

### Summary of Changes

The changes made during the project have been documented: [summary-of-changes/README.md](./summary-of-changes/README.md).

## Apendix

### `docker scout` helloworld output

[Return](#docker-scout-helloworld-example) to `docker scout` helloworld

<details>
   
   <summary> Output from scout should be something similar to this: </summary>

   ```bash
      ✓ Image stored for indexing
      ✓ Indexed 160 packages
      ✓ Provenance obtained from attestation
      ✗ Detected 8 vulnerable packages with a total of 10 vulnerabilities


   ## Overview

                      │         Analyzed Image          
   ───────────────────┼─────────────────────────────────
    Target            │  scout_hello_world_img:latest   
      digest          │  65884f6905ea                   
      platform        │ linux/amd64                     
      vulnerabilities │    0C     0H     2M     8L      
      size            │ 73 MB                           
      packages        │ 160                             


   ## Packages and Vulnerabilities

      0C     0H     1M     0L  pam 1.5.3-5ubuntu5.5
   pkg:deb/ubuntu/pam@1.5.3-5ubuntu5.5?os_distro=noble&os_name=ubuntu&os_version=24.04

      ✗ MEDIUM CVE-2025-8941
         https://scout.docker.com/v/CVE-2025-8941
         Affected range : >=0        
         Fixed version  : not fixed  
      

      0C     0H     1M     0L  tar 1.35+dfsg-3build1
   pkg:deb/ubuntu/tar@1.35%2Bdfsg-3build1?os_distro=noble&os_name=ubuntu&os_version=24.04

      ✗ MEDIUM CVE-2025-45582
         https://scout.docker.com/v/CVE-2025-45582
         Affected range : >=0        
         Fixed version  : not fixed  
      

      0C     0H     0M     3L  curl 8.5.0-2ubuntu10.6
   pkg:deb/ubuntu/curl@8.5.0-2ubuntu10.6?os_distro=noble&os_name=ubuntu&os_version=24.04

      ✗ LOW CVE-2025-9086
         https://scout.docker.com/v/CVE-2025-9086
         Affected range : >=0        
         Fixed version  : not fixed  
      
      ✗ LOW CVE-2025-10148
         https://scout.docker.com/v/CVE-2025-10148
         Affected range : >=0        
         Fixed version  : not fixed  
      
      ✗ LOW CVE-2025-0167
         https://scout.docker.com/v/CVE-2025-0167
         Affected range : >=0        
         Fixed version  : not fixed  
      

      0C     0H     0M     1L  libgcrypt20 1.10.3-2build1
   pkg:deb/ubuntu/libgcrypt20@1.10.3-2build1?os_distro=noble&os_name=ubuntu&os_version=24.04

      ✗ LOW CVE-2024-2236
         https://scout.docker.com/v/CVE-2024-2236
         Affected range : >=0        
         Fixed version  : not fixed  
      

      0C     0H     0M     1L  coreutils 9.4-3ubuntu6.1
   pkg:deb/ubuntu/coreutils@9.4-3ubuntu6.1?os_distro=noble&os_name=ubuntu&os_version=24.04

      ✗ LOW CVE-2016-2781
         https://scout.docker.com/v/CVE-2016-2781
         Affected range : >=0                                           
         Fixed version  : not fixed                                     
         CVSS Score     : 6.5                                           
         CVSS Vector    : CVSS:3.0/AV:L/AC:L/PR:L/UI:N/S:C/C:N/I:H/A:N  
      

      0C     0H     0M     1L  shadow 1:4.13+dfsg1-4ubuntu3.2
   pkg:deb/ubuntu/shadow@1%3A4.13%2Bdfsg1-4ubuntu3.2?os_distro=noble&os_name=ubuntu&os_version=24.04

      ✗ LOW CVE-2024-56433
         https://scout.docker.com/v/CVE-2024-56433
         Affected range : >=0        
         Fixed version  : not fixed  
      

      0C     0H     0M     1L  gnupg2 2.4.4-2ubuntu17.3
   pkg:deb/ubuntu/gnupg2@2.4.4-2ubuntu17.3?os_distro=noble&os_name=ubuntu&os_version=24.04

      ✗ LOW CVE-2022-3219
         https://scout.docker.com/v/CVE-2022-3219
         Affected range : >=0                                           
         Fixed version  : not fixed                                     
         CVSS Score     : 3.3                                           
         CVSS Vector    : CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:N/I:N/A:L  
      

      0C     0H     0M     1L  openssl 3.0.13-0ubuntu3.6
   pkg:deb/ubuntu/openssl@3.0.13-0ubuntu3.6?os_distro=noble&os_name=ubuntu&os_version=24.04

      ✗ LOW CVE-2024-41996
         https://scout.docker.com/v/CVE-2024-41996
         Affected range : >=0        
         Fixed version  : not fixed  
      


   10 vulnerabilities found in 8 packages
   CRITICAL  0  
   HIGH      0  
   MEDIUM    2  
   LOW       8  


   What's next:
      View base image update recommendations → docker scout recommendations scout_hello_world_img:latest


   ```

</details>
