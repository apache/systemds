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

# Install SystemDS from Source

This guide helps in the install and setup of SystemDS from source code.

---

- [1. Install on Windows](#1-install-on-windows)
- [2. Install on Ubuntu](#2-install-on-ubuntu-2204--2404)
- [3. Install on macOS](#3-install-on-macos)
- [4. Build the Project](#4-build-the-project)
- [5. Run a Component Test](#5-run-a-component-test)
- [6. Next Steps](#6-next-steps)

Once the individual environment is set up, you can continue with the common build steps below.

---

# 1. Install on Windows

First setup Java and maven to compile the system note the Java version is 17, we suggest using Java OpenJDK 17.

- <https://openjdk.org/>
- <https://maven.apache.org/download.cgi?.>

Setup your environment variables with JAVA_HOME and MAVEN_HOME. Using these variables add the JAVA_HOME/bin and MAVEN_HOME/bin to the path environment variable. An example of setting it for Java can be found here: <https://www.thewindowsclub.com/set-java_home-in-windows-10>

To run the system we also have to setup some Hadoop and Spark specific libraries. These can be found in the SystemDS repository. To add this, simply take out the files, or add 'src/test/config/hadoop_bin_windows/bin' to PATH. Just like for JAVA_HOME set a HADOOP_HOME to the environment variable without the bin part, and add the `%HADOOP_HOME%\bin` to path.

On Windows, cloning large repositories via GitHub Desktop may stall in some environments. If this happens, cloning via the Git command line is a reliable alternative.
Example:
```bash
git clone https://github.com/apache/systemds.git 
```

To make the build go faster set the IDE or environment variables for Java: '-Xmx16g -Xms16g -Xmn1600m'. Here set the memory to something close to max memory of the device you are using.

To start editing the files remember to import the code style formatting into the IDE, to keep the changes of the files consistent.

A suggested starting point would be to run some of the component tests from your IDE.

# 2. Install on Ubuntu (22.04 / 24.04)

### 2.1 Install Java 17 and Maven

First setup Java, maven and git to compile the system note that the Java version is 17.

```bash
sudo apt update
sudo apt install openjdk-17-jdk maven
sudo apt install -y git
```

Verify the install with:
```bash
java -version
mvn -version
git --version
```

This should return something like:
```bash
openjdk 17.x.x
Apache Maven 3.x.x
git version 2.x.x
```

### 2.2 Set JAVA_HOME for Javadocs

Set `JAVA_HOME` (required for generating Javadocs during the Maven build):
```bash
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))
export PATH="$JAVA_HOME/bin:$PATH"
```

### 2.3 Clone Source Code

Clone the source code:
```bash
cd /opt
git clone https://github.com/apache/systemds.git
cd systemds
```

### 2.4 Testing

R should be installed to run the test suite, since many tests are constructed to compare output with common R packages. One option to install this is to follow the guide on the following link: <https://linuxize.com/post/how-to-install-r-on-ubuntu-20-04/>

R can be installed using the CRAN repository.

**Ubuntu 22.04** 

```bash
sudo apt install dirmngr gnupg apt-transport-https ca-certificates software-properties-common
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'
sudo apt update
sudo apt install r-base
```

**Ubuntu 22.04**

```bash
sudo apt install dirmngr gnupg apt-transport-https ca-certificates software-properties-common
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu noble-cran40/'
sudo apt update
sudo apt install r-base
```

Verify the installation:
```bash
R --version
```

**Install R Dependencies for Integration Tests (Optional)** If you want to run integration tests that depend on additional R packages, install them via:
```bash
Rscript ./src/test/scripts/installDependencies.R
```

# 3. Install on MacOS

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

This should print Java version.

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

# 4. Build the project

To compile the project use in the directory of the source code:
```bash
mvn package -P distribution
```

Example output:
```bash
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  31.730 s
[INFO] Finished at: 2020-06-18T11:00:29+02:00
[INFO] ------------------------------------------------------------------------
```

The first time you package the system it will take longer since maven will download the dependencies. But successive compiles should become faster. The runnable JAR files will appear in `target/`.

### (Optional) Add SystemDS CLI to PATH

After building SystemDS from source, you can add the `bin` directory to your
`PATH` in order to run `systemds` directly from the command line:

```bash
export SYSTEMDS_ROOT=$(pwd)
export PATH="$SYSTEMDS_ROOT/bin:$PATH"
```
This allows you to run `systemds` from the repository root. For running the freshly built executable JAR (e.g., `target/SystemDS.jar`) on Spark, see the Spark section in [Execute SystemDS](run.html).

# 5. Run A Component Test

As an example here is how to run the component matrix tests from command line via maven.

```bash
mvn test -Dtest="**.component.matrix.**"
```

To run other tests simply specify other packages by modifying the test argument part of the command.

# 6. Next Steps

Now everything is setup and ready to go! For running scripts in Spark mode or experimenting with federated workers, see the Execution Guide: [Execute SystemDS](run.html)
