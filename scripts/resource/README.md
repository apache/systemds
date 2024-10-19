# Resource Optimizer
The **Resource Optimizer** is an extension that allows for automatic generation of near optimal cluster configurations for
executing a given SystemDS script in a cloud environment - currently only AWS.
The target execution platform on AWS is EMR (Elastic MapReduce), but single node executions run on EC2 (Elastic Cloud Compute).

## Functionality


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
