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

## Administrator setup


### With [`Cloud Shell`](https://console.aws.amazon.com/cloudshell/home):

Assumed variables,

| Name | Value |
| --- | --- |
| `UserName` | `systemds-bot` |
| `GroupName` | `systemds-group` |

#### 1. Create a user and a group

Create a user and a group, and join user to the created group.

[`create-user`](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/iam/create-user.html)
```sh
[cloudshell-user@host ~]$ aws iam create-user --user-name systemds-bot
{
    "User": {
        "Path": "/",
        "UserName": "systemds-bot",
        "UserId": "AIDAQSHHX7DDAODFXYZ3",
        "Arn": "arn:aws:iam::12345:user/systemds-bot",
        "CreateDate": "2021-04-10T20:36:59+00:00"
    }
}
```

[`create-group`](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/iam/create-group.html)

```sh
[cloudshell-user@host ~]$aws iam create-group --group-name systemds-group
{
    "Group": {
        "Path": "/",
        "GroupName": "systemds-group",
        "GroupId": "AGPAQSHHX7DDB3XYZABCW",
        "Arn": "arn:aws:iam::12345:group/systemds-group",
        "CreateDate": "2021-04-10T20:41:58+00:00"
    }
}
```

#### 2. Attach roles to the group

[`attach-group-policy`](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/iam/attach-group-policy.html)

```sh
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/service-role/AmazonElasticMapReduceRole --group-name systemds-group
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/service-role/AmazonElasticMapReduceforEC2Role --group-name systemds-group
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonElasticMapReduceFullAccess --group-name systemds-group
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AWSKeyManagementServicePowerUser --group-name systemds-group
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/IAMUserSSHKeys --group-name systemds-group

# Grant cloud shell access too.
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AWSCloudShellFullAccess --group-name systemds-group

# To create EC2 keys
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess --group-name systemds-group
```

#### 3. Add user to the group

```sh
aws iam add-user-to-group --user-name systemds-bot --group-name systemds-group
```

#### 4. Create the login-profile with credentials

```sh
$ aws iam create-login-profile --generate-cli-skeleton > login-profile.json
```

`login-profile.json` contains

```json
{
    "LoginProfile": {
        "UserName": "",
        "Password": "",
        "PasswordResetRequired": false
    }
}
```

Create the credentials manually by editing `login-profile.json`.

| Name | Value |
| --- | --- |
| `UserName` | `systemds-bot` |
| `Password` | For example, `9U*tYP` |
| `PasswordResetRequired` | `false` |

Now, create the login profile.

```sh
aws iam create-login-profile --cli-input-json file://login-profile.json
```

---
### With [`AWS CLI`](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html):

1. Create aws account / use your existing aws account

2. Install `aws-cli` specific to your Operating System.

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

## User Setup

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