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

```sh
gcloud dataproc jobs submit spark --cluster ${CLUSTER_NAME} \
  --class org.apache.spark.examples.SparkPi \
  --jars file:///usr/lib/spark/examples/jars/spark-examples.jar -- 1000
```
