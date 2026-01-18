# Install SystemDS from Source

This guide helps in the install and setup of SystemDS from source code.

---

- [1. Install on Windows](#1-install-on-windows)
- [2. Install on Ubuntu 22.04](#2-install-on-ubuntu-2204)
- [3. Install on macOS](#3-install-on-macos)
- [4. Build the Project](#4-build-the-project)
- [5. Run a Component Test](#5-run-a-component-test)
- [6. Next Steps](#6-next-steps)

Once the individual versions is set up skip to the common part of building the system.

---

# 1. Install on Windows

First setup java and maven to compile the system note the java version is 17, we suggest using Java OpenJDK 17.

- <https://openjdk.org/>
- <https://maven.apache.org/download.cgi?.>

Setup your environment variables with JAVA_HOME and MAVEN_HOME. Using these variables add the JAVA_HOME/bin and MAVEN_HOME/bin to the path environment variable. An example of setting it for java can be found here: <https://www.thewindowsclub.com/set-java_home-in-windows-10>

To run the system we also have to setup some Hadoop and spark specific libraries. These can be found in the SystemDS repository. To add this, simply take out the files, or add 'src/test/config/hadoop_bin_windows/bin' to PATH. Just like for JAVA_HOME set a HADOOP_HOME to the environment variable without the bin part, and add the `%HADOOP_HOME%\bin` to path.

On windows, cloning large repositories via GitHub Desktop may stall in some environments. If this happens, cloning via the Git command line is a reliable alternative.
Example:
```bash
git clone https://github.com/apache/systemds.git 
cd systemds
```

Finally if you want to run systemds from command line, add a SYSTEMDS_ROOT that points to the repository root, and add the bin folder to the path.

To make the build go faster set the IDE or environment variables for java: '-Xmx16g -Xms16g -Xmn1600m'. Here set the memory to something close to max memory of the device you are using.

To start editing the files remember to import the code style formatting into the IDE, to keep the changes of the files consistent.

A suggested starting point would be to run some of the component tests from your IDE.

# 2. Install on Ubuntu 22.04

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

R should be installed to run the test suite, since many tests are constructed to compare output with common R packages.
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
This allows you to run `systemds` from the repository root. For running the freshly built executable JAR (e.g., `target/SystemDS.jar`) on Spark, see the Spark section in [Execute SystemDS](run_extended.md).

# 5. Run A Component Test

As an example here is how to run the component matrix tests from command line via maven.

```bash
mvn test -Dtest="**.component.matrix.**"
```

To run other tests simply specify other packages by modifying the test argument part of the command.

# 6. Next Steps

Now everything is setup and ready to go! For running scripts in Spark mode or experimenting with federated workers, see the Execution Guide: [Execute SystemDS](run_extended.md)
