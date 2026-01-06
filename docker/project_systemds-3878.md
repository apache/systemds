# Data Integration and Large-Scale Analysis
## Student Project SYSTEMDS-3878

## Start

1. Fork systemds github on own account
2. Invite project member
1. Create SSH-key locally on computer
1. Add public key to github account
1. Clone project on computer (git clone git@\<link>.git)
1. Get a local version of the systemds project on the computer

### CONTRIBUTING.md

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

#### Example for us:

```text
[MINOR][SYSTEMDS-3878] Fix <file>.Dockerfile vulnerability (<bug>)

This commit fixes the following vulnerability identified by docker scout cves: <...>. The following changes have been made on the Dockerfile: <...>.
<Sources/justifications>
<Shortcomings>
<new epss score if not fully resolved>
```

## Docker scout cves

### Usage 

``` sh
docker scout cves --details --format markdown -o docker/scout_results/sysds_outputX.md --epss apache/systemds
```

To identify tricky packages, using the json output with `--format sarif` helps by showing the path to the vulnerable package.

<https://docs.docker.com/reference/cli/docker/scout/cves/>

1. build systemds project \
   `./docker/build.sh`
   - builds the image: `apache/systemds:latest`
   - default runs: `docker image build -f docker/sysds.Dockerfile -t apache/systemds:latest .`
   - comment out build.sh to change selected `Dockerfile`
2. scout \
   `docker scout cves --details --format markdown -o <docker_subdirecory>/scout_results/<file>_output0.md --epss apache/systemds` (from root dir)
   - `--details`: verbose
   - `--format markdown`: output in markdown format
   - `-o <path_to_file>`: write output to file
   - `-epss`: show epss score
      - `--epss --epss-score 0.1`: filter for vulnerabilities that have more than 10% probability to be exploited.

<details>

   <summary> Directly from filesystem (Dockerfile)</summary>

   > Does Not Work

   > No vulnerability found

   ```bash
   cd project_root_directory
   docker scout cves --format markdown -o scout_results/output0.md --epss fs://docker_subdirecory/file.Dockerfile
   ```

   Add `--epss --epss-score 0.1` to filter for vulnerabilities that have more than 10% probability to be exploited.

</details>

### Solve vulnerabilities

By using the CVE code and reading the description, most vulnerabilities have solutions or workarounds.

#### Upgrading the related package

One way to solve a vulerability is simply to upgrade the package it happens in.

> Sometimes, the package that raised the CVE is not included in the `pom.xml`. This can happen if the package is a dependency of another imported package. \
> `mvn dependency:tree` (output can be found [here](systemds-3878_summary-of-changes.md#output-of-mvn-dependencytree)) shows all imported packages.

### Inspection commands to find packages

[summary-of-changes.md](systemds-3878_summary-of-changes.md#toolbox)

### helloworld example

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


<details>
   
   <summary> Output from scout should be something similar to this: </summary>

   ```bash
      ✓ Image stored for indexing
      ✓ Indexed 160 packages
      ✓ Provenance obtained from attestation
      ✗ Detected 8 vulnerable packages with a total of 10 vulnerabilities


   ## Overview

                     │         Analyzed Image          
   ────────────────────┼─────────────────────────────────
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

## Trivy

## Summary of Changes

Refer to the [documentation](systemds-3878_summary-of-changes.md).