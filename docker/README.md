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
{% end comment %}
-->

# Docker guide

To use docker and systemDS there are two options.
The first is to build your own docker image.
The other option is to download the already build image from docker.

## Build

To Build the docker image simply run the build script.

```bash
./docker/build.sh
```

Afterwards you should have a local image with the id `sebaba/sysds:0.2`.
To execute any given DML script follow the step Run.

## Run

Running SystemDS in a docker container is as simple as constructing any DML script
Then Download the docker image `sebaba/sysds:0.2` or build your own.

```bash
docker pull sebaba/sysds:0.2
```

Verify that the docker image correctly works simply by running it, make sure that your terminal is pointing at the root of you systemds git clone.

```bash
./docker/runDocker.sh
```

This above command will mount the folder `docker/mountFolder`, and execute the script named main.dml inside the folder using the docker container.

You can mount any such folder and execute systemds on by changing the first part of the -v argument of the following command:

```bash
docker run \
  -v $(pwd)/docker/mountFolder:/input \
  --rm sebaba/sysds:0.2
```

## Testing

We also have a docker image for execution of tests.
This enables faster test execution on the github actions.
To build this image simply run the same command as above.

```bash
./docker/build.sh
```

Because the github action pulls the image from docker hub the image has to be pushed to docker hub to produce any change in the behavior of the testing.

```bash
docker push sebaba/testingsysds:0.2
```

For each of the tests that require R, this image is simply used, because it skips the installation of the R packages, since they are installed in this image.

Test your testing image locally by running the following command:

```bash
docker run \
  -v $(pwd):/github/workspace \
  sebaba/testingsysds:0.2 \
  org.apache.sysds.test.component.*.**
```
