---
layout: site
title: SystemDS Install from source
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

This guide helps in the install and setup of SystemDS from source code.

- [Windows Guide](#windows)
- [Ubuntu/Linux Guide](#ubuntu-2004)
- [Mac Guide](#mac)

## Windows

[Developer Guide](windows-source-installation)

---

## Ubuntu 20.04

### Java and Maven

First setup java and maven to compile the system note that the java version is 1.8.

```bash
sudo apt install openjdk-8-jdk-headless
sudo apt install maven
```

Note: To update the `java` command to `openjdk-8` run:
```sh
update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
```

Verify the install with:

```bash
java -version
mvn -version
```

This should return something like:

```bash
openjdk version "1.8.0_252"
OpenJDK Runtime Environment (build 1.8.0_252-8u252-b09-1ubuntu1-b09)
OpenJDK 64-Bit Server VM (build 25.252-b09, mixed mode)
Apache Maven 3.6.3
Maven home: /usr/share/maven
Java version: 1.8.0_252, vendor: Private Build, runtime: /usr/lib/jvm/java-8-openjdk-amd64/jre
Default locale: en_US, platform encoding: UTF-8
OS name: "linux", version: "5.4.0-37-generic", arch: "amd64", family: "unix"
```

### Testing

R is required to be install to run the test suite, since many tests are constructed to comprare output with common R packages.
One option to install this is to follow the guide on the following link: <https://linuxize.com/post/how-to-install-r-on-ubuntu-20-04/>

At the time of writing the commands to install R 4.0.2 are:

```bash
sudo apt install dirmngr gnupg apt-transport-https ca-certificates software-properties-common
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'
sudo apt install r-base
```

Optionally, you need to install the R depedencies for integration tests, like this:
(use `sudo` mode if the script couldn't write to local R library)

```bash
Rscript ./src/test/scripts/installDependencies.R
```

See [Build the project](#Build%20the%20project) to compile the code from here.

---

## MAC

Prerequisite install homebrew on the device.

```bash
# To allow relative paths:
brew install coreutils
# To install open jdk 8.
brew tap adoptopenjdk/openjdk
brew cask install adoptopenjdk8
# Install maven to enable compilation of SystemDS.
brew install maven
```

Then afterwards verify the install:

```bash
java --version
mvn --version
```

This should print something like:

```bash
Java version: 1.8.0_242, vendor: AdoptOpenJDK, runtime: /Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home/jre
Default locale: ru_UA, platform encoding: UTF-8
OS name: "mac os x", version: "10.15.5", arch: "x86_64", family: "mac"

Apache Maven 3.6.3 (cecedd343002696d0abb50b32b541b8a6ba2883f)
Maven home: /usr/local/Cellar/maven/3.6.3_1/libexec
```

Note that if you have multiple __java__ versions installed then you have to change the used version to 8, on __both java and javadoc__. This is done by setting the environment variable JAVA_HOME to the install path of open JDK 8 :

``` bash
export JAVA_HOME=`/usr/libexec/java_home -v 1.8`
```

For running all tests [r-base](https://cran.r-project.org/bin/macosx/) has to be installed as well since this is used as a secondary system to verify the correctness of our code, but it is not a requirement to enable building the project.

Optionally, you need to install the R depedencies for integration tests, like this:
(use `sudo` mode if the script couldn't write to local R library)

```bash
Rscript ./src/test/scripts/installDependencies.R
```

See [Build the project](#Build%20the%20project) to compile the code from here.

---

## Build the project

To compile the project use:

```bash
mvn package -P distribution
```

```bash
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  31.730 s
[INFO] Finished at: 2020-06-18T11:00:29+02:00
[INFO] ------------------------------------------------------------------------
```

The first time you package the system it will take longer since maven will download the dependencies.
But successive compiles should become faster.

Now everything is setup and ready to go!
To execute dml scripts look at [Execute SystemDS](run)
