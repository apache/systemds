---
layout: site
title: Use SystemDS with Docker
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


[Docker](https://docs.docker.com/get-docker/) enables you to separate applications from
your infrastructure. This provides a way to manage the infrastructure the same way
you do with the software.

With Docker, enabling GPU support would be much easier on linux. Since only the NVIDIA
GPU drivers are required on the host machine (NVIDIA CUDA toolkit is not required).

## SystemDS Docker requirements

Install [Docker](https://docs.docker.com/get-docker/) specific to your machine

Note: If you would like to manage docker as a non-root user, refer to
[linux-postinstall](https://docs.docker.com/engine/install/linux-postinstall/)

## Download SystemDS Docker image

The official SystemDS docker images are located at [apache/systemds](https://hub.docker.com/r/apache/systemds)
Docker Hub repository. Image releases are tagged based on the release channel:

| Tag | Description |
| --- | --- |
| `nightly` | Builds for SystemDS `main` branch. Used by SystemDS developers |


Usage examples:

```sh
docker pull apache/systemds:nightly         # Nightly release with CPU
```

### Start the Docker container


Options:

- `-it` - interactive
- `--rm` - cleanup
- `-p` - port forwarding

For comprehensive guide, refer [`docker run`](https://docs.docker.com/engine/reference/run/)

```sh

docker run [-it] [--rm] [-p hostPort:containerPort] apache/systemds[:tag] [command]
```

#### Examples

To verify the SystemDS installation,

Create a `dml` file, for example

```sh
touch hello.dml

cat <<EOF >>./hello.dml
print("This is SystemDS")
EOF
```
and run it.

```sh
docker run -it --rm -v $PWD:/tmp -w /tmp apache/systemds:nightly systemds ./hello.dml
```

The output is `"This is SystemDS"` after successful installation.
For SystemDS usage instructions, see [standalone instructions](./run).


This way you can run a DML program developed on the host machine, mount the host directory and change the
working directory with [`-v` flag](https://docs.docker.com/engine/reference/run/#volume-shared-filesystems)
and [`-w` flags](https://docs.docker.com/engine/reference/run/#workdir).


