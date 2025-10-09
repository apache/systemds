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

Afterwards you should have a local image with the id `apache/systemds:latest`.
To execute any given DML script follow the step Run.

## Run

Running SystemDS in a docker container is as simple as constructing any DML script.
Then download the docker image `apache/systemds:latest` or build your own.

```bash
docker pull apache/systemds:latest
```

Verify that the docker image correctly works simply by running it.

```bash
docker run --rm apache/systemds:latest
```

It should respond with something like:

```txt
Hello, World!
SystemDS Statistics:
Total execution time:  0.010 sec.
```

To run specific scripts mount a folder(s) containing the scripts and data,
and execute the script inside the folder using the docker container.

You can mount any such folder and execute systemds by changing the first part of the -v argument of the following command:

```bash
docker run \
  -v $(pwd)/docker/mountFolder:/input \
  --rm apache/systemds:latest
```

Default behavior is to run the script located at /input/main.dml.
To run any other script use:

```bash
docker run \
  -v $(pwd)/folder/to/share:/any/path/in/docker/instance \
  --rm apache/systemds:latest \
  systemds /any/path/to/a/script.dml
```

### Docker run worker node

To run a federated worker in a docker container simply use:

```bash
docker run -p 8000:8000 --rm apache/systemds:latest systemds WORKER 8000
```

This port forwards the worker to port 8000 on the host and starts a worker in the instance on port 8000.

Note that the worker does not have any data, since no data is mounted in the worker image.
To add a folder containing the data needed in the worker do the following:

```bash
docker run \
  -p 8000:8000 \
  -v $(pwd)/data/folder/path/locally:/data/folder/path/in/container \
  --rm apache/systemds:latest systemds WORKER 8000
```

### Docker run python script

To run a python script the `pythonsysds` image is used.

```bash
docker run --rm apache/systemds:python-nightly
```

User provided scripts have to be mounted into the image.

```bash
docker run \
  -v $(pwd)/data/folder/path/locally:/data/folder/path/in/container \
  --rm apache/systemds:latest \
  python3 path/to/script/to/execute.py
```

## Testing image

We also have a docker image for execution of tests.
This enables faster test execution on the github actions.
To build this image simply run the same command as above.

```bash
./docker/build.sh
```

Because the github action pulls the image from docker hub the image has to be pushed to docker hub to produce any change in the behavior of the testing.

```bash
docker push apache/systemds:testing-latest
```

For each of the tests that require R, this image is simply used, because it skips the installation of the R packages, since they are installed in this image.

Test your testing image locally by running the following command:

```bash
docker run \
  -v $(pwd):/github/workspace \
  -v $HOME/.m2/repository:/root/.m2/repository \
  apache/systemds:testing-latest \
  org.apache.sysds.test.component.**
```
