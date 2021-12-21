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

## Create a dataproc cluster

Create a cluster name
```sh
CLUSTERNAME=dp-systemds
```

Set Dataproc cluster region
```sh
gcloud config set dataproc/region us-central1
```

Now, create a new cluster

[`gcloud dataproc clusters create` reference](https://cloud.google.com/sdk/gcloud/reference/dataproc/clusters/create)
```sh
gcloud dataproc clusters create ${CLUSTERNAME} \
  --scopes=cloud-platform \
  --tags systemds \
  --zone=us-central1-c \
  --worker-machine-type n1-standard-2 \
  --worker-boot-disk-size 500 \
  --master-machine-type n1-standard-2 \
  --master-boot-disk-size 500 \
  --image-version 2.0
```

## Submit a Spark job to the cluster

Jobs can be submitted via a Cloud Dataproc API
[`jobs.submit`](https://cloud.google.com/dataproc/docs/reference/rest/v1/projects.regions.jobs/submit) request

Submit an example job using `gcloud` tool from the Cloud Shell command line

Test that the cluster is setup properly:

```sh
gcloud dataproc jobs submit spark --cluster ${CLUSTERNAME} \
  --class org.apache.spark.examples.SparkPi \
  --jars file:///usr/lib/spark/examples/jars/spark-examples.jar -- 1000
```

### Add SystemDS library to the cluster

SSH into the cluster, download the artifacts from https://dlcdn.apache.org/systemds/
and copy jar file in the `lib` folder.

```sh
gcloud compute ssh ${CLUSTERNAME}-m --zone=us-central1-c
wget https://dlcdn.apache.org/systemds/2.2.0/systemds-2.2.0-bin.zip
unzip -q systemds-2.2.0-bin.zip
mkdir /usr/lib/systemds
cp systemds-2.2.0-bin/systemds-2.2.0.jar /usr/lib/systemds
```

### Run SystemDS as a Spark job

```sh
gcloud dataproc jobs submit spark --cluster ${CLUSTERNAME} \
  --class org.apache.sysds.api.DMLScript \
  --jars file:///usr/lib/systemds/systemds-2.2.0.jar -- 1000
```

### Job info and connect

List all the jobs:

```sh
gcloud dataproc jobs list --cluster ${CLUSTERNAME}
```

To get output of a specific job note `jobID` and in the below command
replace `jobID`.

```sh
gcloud dataproc jobs wait jobID
```

### Resizing the cluster

For intensive computations, to add more nodes to the cluster either to speed up.

Existing cluster configuration

```sh
gcloud dataproc clusters describe ${CLUSTERNAME}
```

Add preemptible nodes to increase cluster size:

```sh
gcloud dataproc clusters update ${CLUSTERNAME} --num-preemptible-workers=1
```

Note: `workerConfig` and `secondaryWorkerConfig` will be present.

### SSH into the cluster

SSH into the cluster (primary node) would provide fine grained control of the cluster.

```sh
gcloud compute ssh ${CLUSTERNAME}-m --zone=us-central1-c
```

Note: For the first time, we run `ssh` command on Cloud Shell, it will generate SSH keys
for your account.

The `--scopes=cloud-platform` would allow us to run gcloud inside the cluster too.
For example,

```sh
gcloud dataproc clusters list --region=us-central1
```

to exit the cluster primary instance

```sh
logout
```

### Deleting the cluster

```
gcloud dataproc clusters delete ${CLUSTERNAME}
```

### Tags

A `--tags` option allows us to add a tag to each node in the cluster. Firewall rules
can be applied to each node with conditionally adding flags.

