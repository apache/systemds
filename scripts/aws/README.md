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

Instructions:

1. Create aws account / use your existing aws account

2. Install aws-cli on your system 

(https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html)

3. Create a user
    
    * Create a new user (https://console.aws.amazon.com/iam/home?#/users)

    * Create new group and add the following policies to it:
         
         - AmazonElasticMapReduceRole
         
         - AmazonElasticMapReduceforEC2Role
         
         - AdministratorAccess
         
         - AmazonElasticMapReduceFullAccess
         
         - AWSKeyManagementServicePowerUser
         
         - IAMUserSSHKeys 

4. Configure your aws-cli (https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html#cli-quick-configuration)

5. Spin up an EMR cluster with SystemDS
    
    * Put your SystemDS artifacts (dml-scripts, jars, config-file) in the directory systemds 
    
    * Edit configuration in: systemds_cluster.config
    
    * Run: ./spinup_systemds_cluster.sh
    
6. Run a SystemDS script
    
    * Run: ./run_systemds_script.sh path/to/script.dml 
         With args: ./run_systemds_script.sh path/to/script.dml "1.0, 2.6"  
    
7. Terminate the EMR cluster: ./terminate_systemds_cluster.sh
    
#### Further work

* Finetune the memory 
    
    https://aws.amazon.com/blogs/big-data/best-practices-for-successfully-managing-memory-for-apache-spark-applications-on-amazon-emr/
    https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark-configure.html#spark-defaults
* Test if Scale to 100 nodes

* Make the cluster WebUIs (Ganglia, SparkUI,..) accessible from outside

* Integrate spot up instances 