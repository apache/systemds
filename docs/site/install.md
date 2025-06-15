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

- [Install](#install)
  - [Windows](#windows)
  - [Ubuntu 22.04](#ubuntu-2204)
    - [Testing](#testing)
  - [MAC](#mac)
- [2. Build the project](#2-build-the-project)
- [3. Run A Component Test](#3-run-a-component-test)

Once the individual versions is set up skip to the common part of building the system.

---

## Install

---

### Windows

First setup java and maven to compile the system note the java version is 17, we suggest using Java OpenJDK 17.

- <https://openjdk.org/>
- <https://maven.apache.org/download.cgi?.>

Setup your environment variables with JAVA_HOME and MAVEN_HOME. Using these variables add the JAVA_HOME/bin and MAVEN_HOME/bin to the path environment variable. An example of setting it for java can be found here: <https://www.thewindowsclub.com/set-java_home-in-windows-10>

To run the system we also have to setup some Hadoop and spark specific libraries. These can be found in the SystemDS repository. To add this, simply take out the files, or add 'src/test/config/hadoop_bin_windows/bin' to PATH. Just like for JAVA_HOME set a HADOOP_HOME to the environment variable without the bin part, and add the %HADOOP_HOME%/bin to path.

Finally if you want to run systemds from command line, add a SYSTEMDS_ROOT that points to the repository root, and add the bin folder to the path.

To make the build go faster set the IDE or environment variables for java: '-Xmx16g -Xms16g -Xmn1600m'. Here set the memory to something close to max memory of the device you are using.

To start editing the files remember to import the code style formatting into the IDE, to keep the changes of the files consistent.

A suggested starting point would be to run some of the component tests from your IDE.

---

### Ubuntu 22.04

First setup java and maven to compile the system note that the java version is 17.

```bash
sudo apt install openjdk-17-jdk
sudo apt install maven
```

Verify the install with:

```bash
java -version
mvn -version
```

This should return something like:

```bash
openjdk 17.0.11 2024-04-16
OpenJDK Runtime Environment Temurin-17.0.11+9 (build 17.0.11+9)
OpenJDK 64-Bit Server VM Temurin-17.0.11+9 (build 17.0.11+9, mixed mode, sharing)

Apache Maven 3.9.9 (8e8579a9e76f7d015ee5ec7bfcdc97d260186937)
Maven home: /home/usr/Programs/maven
Java version: 17.0.11, vendor: Eclipse Adoptium, runtime: /home/usr/Programs/jdk-17.0.11+9
Default locale: en_US, platform encoding: UTF-8
OS name: "linux", version: "6.8.0-59-generic", arch: "amd64", family: "unix"
```

#### Testing

R is required to be install to run the test suite, since many tests are constructed to compare output with common R packages.
One option to install this is to follow the guide on the following link: <https://linuxize.com/post/how-to-install-r-on-ubuntu-20-04/>

At the time of writing the commands to install R 4.0.2 are:

```bash
sudo apt install dirmngr gnupg apt-transport-https ca-certificates software-properties-common
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'
sudo apt install r-base
```

Optionally, you need to install the R dependencies for integration tests, like this:
(use `sudo` mode if the script couldn't write to local R library)

```bash
Rscript ./src/test/scripts/installDependencies.R
```

---

### MAC

Prerequisite install homebrew on the device.

```bash
# To allow relative paths:
brew install coreutils
# To install open jdk 17.
brew install openjdk@17
# Install maven to enable compilation of SystemDS.
brew install maven
```

Then afterwards verify the install:

```bash
java --version
mvn --version
```

This should print java version.

Note that if you have multiple __java__ versions installed then you have to change the used version to 17, on __both java and javadoc__. This is done by setting the environment variable JAVA_HOME to the install path of open JDK 17 :

```bash
export JAVA_HOME=`/usr/libexec/java_home -v 17`
```

For running all tests [r-base](https://cran.r-project.org/bin/macosx/) has to be installed as well since this is used as a secondary system to verify the correctness of our code, but it is not a requirement to enable building the project.

Optionally, you need to install the R dependencies for integration tests, like this:
(use `sudo` mode if the script couldn't write to local R library)

```bash
Rscript ./src/test/scripts/installDependencies.R
```

---

## 2. Build the project

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

## 3. Run A Component Test

As an example here is how to run the component matrix tests from command line via maven.

```bash
mvn test -Dtest="**.component.matrix.**"
```

To run other tests simply specify other packages by modifying the
test argument part of the command.

Now everything is setup and ready to go!

To execute dml scripts look at [Execute SystemDS](run), this step is not needed to develop in systemds, but it helps setting up the command-line execution of systemds.
