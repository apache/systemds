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

## Developing Apache SystemDS on Windows Platform

These instructions will help you build Apache SystemDS from source code, which is the basis for the engine
and algorithms development. The following conventions will be used to refer to directories on your machine:

* `<USER_HOME>` is your home directory.
* `<JDK_18_HOME>` is the root directory for the 1.8 JDK.
* `<MAVEN_HOME>` is the root directory for the Apache Maven source code.
* `<SYSTEMDS_HOME>` is the root directory for the SystemDS source code.
* `<SPARK_HOME>` is the root directory for the Apache Spark source code.
* `<HADOOP_HOME>` is the root directory for 
* `<CUDA_HOME>`,`<CUDA_PATH>` is the top directory for NVIDIA GPU Computing Toolkit.  
Ex. For version `9.0`, it would like `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0`
Also, make sure that `libnvvp`, `bin` folders are accessible to command line.
*  `<CUDNN_PATH>` is the path 
Ex. Corresponding to the CUDA version, `C:\Program Files\cudnn-9.0-windows10-x64-v7.2.1.38\cuda`

#### Directory structure of the installation
```py
.
├── MAVEN_HOME = './maven'
|   ├── bin
|       ├── mvn.exe # and other executables
├── SPARK_HOME = './spark'
|   ├── bin
|       ├── spark-shell # and other invocation commands
├── HADOOP_HOME = './hadoop'
│   ├── bin
|       ├── wintuils.exe # The Default content layout and html file.
├── SYSTEMDS_HOME

```

### Getting Apache SystemDS Source Code

SystemDS source code is available from [github.com/apache/systemds](https://github.com/apache/systemds) by either cloning or
downloading a zip file (based on a branch) into `<SYSTEMDS_HOME>`. The default is the *master* branch.

````
git clone https://github.com/apache/systemds systemds
```` 

The master branch contains the source code which will be used to create the next major version of Apache SystemDS.

_**Speed Tip:**_ If the complete repository history isn't needed then using a shallow clone (`git clone --depth 1`) will
save significant time.

### Building SystemDS source code

`IntelliJ IDEA` or `Eclipse` is preferred for best developer experience.

#### Opening the IntelliJ Source Code for Build

Using IntelliJ IDEA **File | Open**, select the `<SYSTEMDS_HOME>` directory. 
* If IntelliJ IDEA displays an error about a missing or out of date required plugin (e.g. maven),
  [enable, upgrade, or install that plugin](https://www.jetbrains.com/help/idea/managing-plugins.html) and restart IntelliJ IDEA.

#### IntelliJ Build Configuration

JDK version 1.8 (u151 or newer) is required for building and developing for SystemDS developement.

1. Using IntelliJ IDEA, [configure](https://www.jetbrains.com/help/idea/sdk.html) a JDK named "**1.8**", pointing to `<JDK_18_HOME>`.
   * If not already present, add `<JDK_18_HOME>/lib/tools.jar` [to the Classpath](https://www.jetbrains.com/help/idea/sdk.html#manage_sdks) tab
     for the **1.8** JDK.
2. If the _Maven Integration_ plugin is disabled, [add the path variable](https://www.jetbrains.com/help/idea/working-with-projects.html#path-variables)
   "**MAVEN_REPOSITORY**" pointing to `<USER_HOME>/.m2/repository` directory.
3. _**Speed Tip:**_ If you have enough RAM on your computer,
   [configure the compiler settings](https://www.jetbrains.com/help/idea/specifying-compilation-settings.html)
   to enable the "Compile independent modules in parallel" option. Also set the "User-local build process VM options" to `-Xmx2G`.
   These changes will greatly reduce the compile time.
4. Now, selecting the IntelliJ IDEA **Build | Build module 'systemds'** option starts the maven build.
5. _**Speed Tip:**_ If the development machine have enough RAM,
   [configure the compiler settings](https://www.jetbrains.com/help/idea/specifying-compilation-settings.html)
   to enable the "Compile independent modules in parallel" option.
 
#### Building the Source Code
To build SystemDS from source, choose **Build | Build Project** from the main menu.
OR
To maven build, run the `mvn clean package` command in `<SYSTEMDS_HOME>` directory. See the `pom.xml` file for details.

### Testing
To run the SystemDS built from source, choose **Run | Run** from the main menu.

To run tests on the build, apply these setting to the **Run | Edit Configurations... | Defaults | JUnit** configuration tab:
  * Working dir: `<SYSTEMDS_HOME>`
  * VM options:
    * `-ea`
