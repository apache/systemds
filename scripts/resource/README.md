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

# Resource Optimizer
The **Resource Optimizer** is an extension that allows for automatic generation of (near) optimal cluster configurations for
executing a given SystemDS script in a cloud environment - currently only AWS.
The target execution platform on AWS is EMR (Elastic MapReduce), but single node executions run on EC2 (Elastic Cloud Compute). In both cases files are expected to be pulled from Amazon S3 (Simple Storage Service).

## Functionality

The Resource Optimizer is extension of SystemDS and while employing the systems' general tools for program compilation and execution, this extension does not operate in execution time but is rather a separate executable program meant to be used upon running a DML script (program) in the cloud (AWS). The extension runs an optimization algorithm for finding the optimal cluster configurations for executing the target SystemDS program considering the following properties:

* The input dataset properties (providing metadata files is required)
* A defined set of available machine type for running the program in the cloud  
* The resulting execution plan of the program for the corresponding combination of dataset characteristics and hardware characteristics
* The optimization objective: user-defined goal of optimization with options to aim for shortest execution time, lowest monetary cost or (by default) for configuration set fairly balancing both objectives. 

The result is a set of configuration files with all options needed for launching SystemDS in the cloud environment. 

For complete the automation process of running SystemDS program in the cloud, the extension is complimented with corresponding scripts to allow the user to launch the program with the generated optimal configurations with minimal additional effort.

## User Manual

### Installation
The extension, although not part of the execution time of SystemDS, is fully integrated with the project and is compiled 
as part of the general compilation process using Maven. Check [here](../../README.md) for more information about that.

Maven compiles a JAR file which is to be used for running the extension. 
The path to the file is `<SystemDS root folder>\target\ResourceOptimizer.jar`, 
but for easier usage you can complete the following steps from the project root folder to configure it on path:
```shell
# general step for SystemDS
export SYSTEMDS_ROOT=$(pwd)
# specific steps for the extension
export PATH=$PATH:$SYSTEMDS_ROOT/scripts/resource/bin
```
The proper execution requires JDK 11 so make sure to export the correct JDK version to `$JAVA_HOME` environmental variable.

### Usage

The extension is installed as a separate java executable files, so it can be launched with `java -jar <SystemDS root folder>\target\ResourceOptimizer.jar ...`, but if you completed all additional steps from the **Installation** section then you can run directly the extension using the `systemds-ropt` command globally without specifying a target jar-file.

The executable takes the following arguments:
```txt
 -args <argN>          specifies positional parameters; first value will replace $1 in DML program, $2 will replace 2nd and so on

 -f <filename>         specifies DML file to execute; path should be local

 -help                 shows usage message

 -nvargs <key=value>   parameterizes DML script with named parameters of the form <key=value>; <key> should be a valid identifier in DML

 -options <arg>        specifies options file for the resource optimization

```
Either `-help` or `-f` is required, where `-f` lead to the actual program execution. In that case if `-options` is not provided, that program will look for a default `options.properties` in the current directory and if no file found the program will fail. Like for SystemDS program execution, `-args` and `-nvargs` provide additional script arguments and they cannot be specified simultaneously. The `-options` arguments point to a file with options to customize further the optimization process. These options provide paths to further properties files and a optional optimization configurations and constraints.

It is important for program arguments being dataset filepaths to be specified with their URI address that will be actually used by the SystemDS program later. Currently the only supported and tested option is S3 for storing input and outputs on AWS. Check [Required Options](#required-options) for further details.

## Providing Options

Although automatic, the optimization process requires providing certain properties beyond the SystemDS program and the dataset characteristics. These are in general hardware characteristics for potential resources available on the target cloud platform. In addition to that, the user can (and in some cases should) provide a further set of constraints options for to limit ensure that the resulting configuration would be feasible for the target scenario. All properties and constraints are provided with the options file (using the `-options` argument) and a full set all possible options can be found in the default options file: `./scripts/resource/options.properties`

### Required Options

For the proper operation certain options should be always provided:

* `REGION` - AWS region which is crucial for the monetary estimations
* `INFO_TABLE` - a `.csv` table providing the needed hardware and price characteristics for each of the potential server machines (instances) for consideration
* `REGION_TABLE` - another `.csv` table providing additional pricing parameters (EMR fee, EBS Storage prices)
* `OUTPUT_FOLDER` - path for the resulting configuration files

Depending on the chosen optimization objective method further options could be required:
* `MAX_TIME` - in case of choosing optimizing for monetary cost only
* `MAX_PRICE` - in case of choosing optimizing for execution time only

The enumeration strategy and optimization functions has defaults values and are not required to be further set in the options:
* Default for enumeration strategy is: grid
* Default for optimization function is: costs

We provide a table comprising the relevant EC2 instances characteristics supported currently by the resource optimizer 
and table with pricing parameters for all regions supported by EMR. The python script from `update_prcies.py` provides and automation
for updating the prices of the EC2 instances in the fist mention table be setting a target AWS region.

As mentioned in [Usage](#usage), the filepath s for the input datasets should be the URI addresses for the distributed S3 file. This allows Resource Optimizer to account for the costs of fetching these external files. To allow greater flexibility at using the extension, the user is provided with the possibility of omitting this requirement in certain scenarios: the user provides the program arguments as S3 URI paths but with an imaginary name (and leading S3 URI schema - `s3://`) and then fills the `LOCAL_INPUTS` property. This property holds key-value pairs where the key is the imaginary S3 file paths and the value is the local path for these files. The local path could also point to a non existing file as long as a corresponding metadata (`.mtd`) file is locally available on this path. 

### Further Options 

As mentioned above, the user can decide to switch between different options for the enumeration strategy (option `ENUMERATION`) and the optimization functions (options `OPTIMIZATION_FUNCTION`). 

The enumeration strategy has influence mostly on the speed of completing the process of finding the optimal configuration. The default value is `grid` and this sets a grid-based enumeration where each configuration combination within the configured constraints is being evaluated. The next possible option value is `prune` which prunes dynamically certain configuration during the progress of the enumeration process based on the intermediate results for the already evaluated configurations. In theory this should deliver the same optimal result as the grid-based enumeration and the experiments so far proved that while showed great speed-up. The last possibility is fot that option is `interest`, which uses a interest-based enumeration which uses several (configurable) criterias for statically reducing the search space for the optimal configuration base on program and hardware properties.

Here is a list of all the rest of the options available:

* `CPU_QUOTA` (default 1152) - specifies the limit of (virtual) CPU cores allowed for evaluation. This corresponds to the EC2 service quota for limiting the running instances within the same region at the same moment.
* `COSTS_WEIGHT` (default 0.01) - specifies the weighing factor for the multi-objective function for optimization
* `MIN_EXECUTORS` (default 0) - specifies minimum desired executors, where 0 includes single node execution. Allows configuring minimum cluster size.
* `MAX_EXECUTORS` (default 200) - specifies maximum desired executors. The maximum number of executors can be limited dynamically further more in case of reaching the CPU quota number.
* `INSTANCE_FAMILIES` - specifies VM instance types for consideration at searching for optimal configuration. If not specified, all instances from the table with instance metadata are considered
* `INSTANCE_SIZES` - specifies VM instance sizes for consideration at searching for optimal configuration. If not specified, all instances from the table with instance metadata are considered 
* `STEP_SIZE` (default 1) - specific to grid-based enumeration strategy: specifies step size for enumerating number of executors
* `EXPONENTIAL_BASE` - specific to grid-based enumeration strategy: specifies exponential base for increasing the number of executors exponentially if a value greater than 1 given
* `USE_LARGEST_ESTIMATE` (default *true*) - specific to the enumeration strategy with interest criterias: boolean ('true'/'false') to indicate if single node execution should be considered only in case of sufficient memory budget for the driver
* `USE_CP_ESTIMATES` (default *true*) - specific to the enumeration strategy with interest criterias: boolean ('true'/'false') to indicate if the CP memory is an interest for the enumeration
* `USE_BROADCASTS` (default *true*) - specific to the enumeration strategy with interest criterias: boolean ('true'/'false') to indicate if potential broadcast variables' sizes is an interest for driver and executors memory budget
* `USE_OUTPUTS` (default *false*) - specific to the enumeration strategy with interest criterias: boolean ('true'/'false') to indicate if the size of the outputs (potentially cached) is an interest for the enumerated number of executors. False by default since the caching process is not considered by the current version of the Resource Optimizer.

## Launching Program in the Cloud

For optimal solution with single node, the target environment in AWS is an EC2 instance and the resulting file from the optimization process is a single file: `ec2_configurations.json`. This files store the all values required for launching an EC2 instance and running SystemDS on it. The file has a custom format is not supported by AWS CLI.

For optimal solution with Spark cluster, the target environment is EMR cluster. The Resource optimizer generates in this case two files: 
* `emr_instance_groups.json` - contains hardware configurations for machines in the cluster
* `emr_configurations.json` - contains Spark-specific configurations for the cluster

Both files follow a structure supported by AWS CLI so they can be directly passed in a command for launching an EMR cluster in combination with all further required options.

*Note: For proper execution of all of the shell scripts mentioned bellow, please do not delete any of the optional variables in the corresponding `.env` files but just leave them empty, otherwise the scripts could fail due to "unbound variable" error*

### Launching in Single-node Mode Automatically

For saving additional user effort and the need of background knowledge for Amazon EC2, the project includes scripts for automating the processes of launching an instance based on the output configurations and running the target SystemDS script on it. 

The shell script `./scripts/resource/launch/single_node_launch.sh` is used for launching the EC2 instance based in a corresponding `ec2_configurations.json`. All configuration and options should be specified in `./scripts/resource/launch/single_node.env` and the script expects no arguments passed. The options file includes all required and optional configurations and explanations for their functionality. The launch process utilizes AWS CLI and executes automatically the following steps:
1. Query the AWS API for an OS image with Ubuntu 24.04 for the corresponding processor architecture and AWS region.
2. Generates instance profile for accessing S3 by the instance
3. Launches the target EC2 with all additional options needed in our case and with providing a bootstrap script for SystemDS installation
4. Waits for the instance to enter state `RUNNING` 
5. Waits for the completion of the SystemDS installation

*Note: the SSH key generation is not automated and an existing key should be provided in the options file before executing the scripts, otherwise the launch will fail.*

After the script completes without any errors, the instance is already fully prepared for running SystemDS programs. However, before running the script for automated program execution, the user needs to manually uploads the needed files (including the DML script) to S3.

Once all files are uploaded to S3, the user can execute the `./scripts/resource/launch/single_node_run_script.sh` script. All configurations are again provided via the same options file like for the launch. This scripts does the following:
1. Prepare a command for executing the SystemDS program with all the required arguments and with optimal JVM configurations.
2. Submits the command to the target machine and additionally sets simple a logging mechanism: the execution writes directly to log files that are compressed after program completion/fail to S3.
3. Optionally (depending on the option `AUTO_TERMINATION`) the scripts sets that the instance should be stopped after the program completes/fails and the log files are uploaded to S3. In that case the script wait for the machine to enters state `STOPPED` and trigger its termination.

The provided URI addresses for S3 files should always use the `s3a://` prefix to allow for the proper functionality
of the Hadoop-AWS S3 connector. 

*Note 1: if automatic termination is disabled the user should manually check for program completion and terminate the EC2 instance*

*Note 2: if automatic termination is enabled the user should ensure that all the output files are written to S3 because the EC2 instance storage is always configured to be ephemeral.*

### Launching in Cluster (Hybrid) Mode Automatically 

The project includes also the equivalent script files to automate the launching process of EMR cluster 
and the submission of steps for executing SystemDS programs.

The shell script `./scripts/resource/launch/cluster_launch.sh` is used ofr launching the cluster 
based on the auto-generated files from the Resource Optimizer. Additional configurations regarding the launching
process or submitting steps should be defined in `./scripts/resource/launch/cluster.env` and the script does not
expect any passed arguments. Like for EC2 launch, the script uses AWC CLI and executed the following steps:
1. Queries the default subnet in the user have not defined one
2. In case of provided SystemDS script for execution in the configuration file, it prepares the whole step definition
3. Launched the cluster with all provided configurations. Depending on the set value for `AUTO_TERMINATION_TIME` the cluster can be set 
to be automatically terminated after the completion of the initially provided step (of one provided at all) or terminate automatically 
after staying for a given period of time in idle state.
4. The script waits until the cluster enter state `RUNNING` and completes.

The script `./scripts/resource/launch/cluster_run_script.sh` can be used for submitting
steps to running EMR cluster, again by getting all its arguments from the file `./scripts/resource/launch/cluster.env`.
The script will for the completion of the step by polling for the step's state and 
if `AUTO_TERMINATION_TIME` is set to 0 the cluster will be automatically terminated.


The provided URI addresses for S3 files should always use the `s3://` prefix to allow for the proper functionality
of the EMR-specific S3 connector. 

*The same notes as for the launch on programs as EC2 are valid here as well!*
