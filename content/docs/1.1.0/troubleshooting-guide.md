---
layout: global
title: Troubleshooting Guide
description: Troubleshooting Guide
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

* This will become a table of contents (this text will be scraped).
{:toc}

<br/>


## ClassNotFoundException for commons-math3

The Apache Commons Math library is utilized by SystemML. The commons-math3
dependency is included with Spark and with newer versions of Hadoop. Running
SystemML on an older Hadoop cluster can potentially generate an error such
as the following due to the missing commons-math3 dependency:

	java.lang.ClassNotFoundException: org.apache.commons.math3.linear.RealMatrix

This issue can be fixed by changing the commons-math3 `scope` in the pom.xml file
from `provided` to `compile`.

	<dependency>
		<groupId>org.apache.commons</groupId>
		<artifactId>commons-math3</artifactId>
		<version>3.1.1</version>
		<scope>compile</scope>
	</dependency>

SystemML can then be rebuilt with the `commons-math3` dependency using
Maven (`mvn clean package -P distribution`).

## OutOfMemoryError in Hadoop Reduce Phase 
In Hadoop MapReduce, outputs from mapper nodes are copied to reducer nodes and then sorted (known as the *shuffle* phase) before being consumed by reducers. The shuffle phase utilizes several buffers that share memory space with other MapReduce tasks, which will throw an `OutOfMemoryError` if the shuffle buffers take too much space: 

    Error: java.lang.OutOfMemoryError: Java heap space
        at org.apache.hadoop.mapred.IFile$Reader.readNextBlock(IFile.java:357)
        at org.apache.hadoop.mapred.IFile$Reader.next(IFile.java:419)
        at org.apache.hadoop.mapred.Merger$Segment.next(Merger.java:238)
        at org.apache.hadoop.mapred.Merger$MergeQueue.adjustPriorityQueue(Merger.java:348)
        at org.apache.hadoop.mapred.Merger$MergeQueue.next(Merger.java:368)
        at org.apache.hadoop.mapred.Merger.writeFile(Merger.java:156)
        ...
  
One way to fix this issue is lowering the following buffer thresholds.

    mapred.job.shuffle.input.buffer.percent # default 0.70; try 0.20 
    mapred.job.shuffle.merge.percent # default 0.66; try 0.20
    mapred.job.reduce.input.buffer.percent # default 0.0; keep 0.0

These configurations can be modified **globally** by inserting/modifying the following in `mapred-site.xml`.

    <property>
     <name>mapred.job.shuffle.input.buffer.percent</name>
     <value>0.2</value>
    </property>
    <property>
     <name>mapred.job.shuffle.merge.percent</name>
     <value>0.2</value>
    </property>
    <property>
     <name>mapred.job.reduce.input.buffer.percent</name>
     <value>0.0</value>
    </property>

They can also be configured on a **per SystemML-task basis** by inserting the following in `SystemML-config.xml`.

    <mapred.job.shuffle.merge.percent>0.2</mapred.job.shuffle.merge.percent>
    <mapred.job.shuffle.input.buffer.percent>0.2</mapred.job.shuffle.input.buffer.percent>
    <mapred.job.reduce.input.buffer.percent>0</mapred.job.reduce.input.buffer.percent>

Note: The default `SystemML-config.xml` is located in `<path to SystemML root>/conf/`. It is passed to SystemML using the `-config` argument:

    hadoop jar SystemML.jar [-? | -help | -f <filename>] (-config <config_filename>) ([-args | -nvargs] <args-list>)
    
See [Invoking SystemML in Hadoop Batch Mode](hadoop-batch-mode.html) for details of the syntax. 

## Total size of serialized results is bigger than spark.driver.maxResultSize

Spark aborts a job if the estimated result size of collect is greater than maxResultSize to avoid out-of-memory errors in driver.
However, SystemML's optimizer has estimates the memory required for each operator and provides guards against these out-of-memory errors in driver.
So, we recommend setting the configuration `--conf spark.driver.maxResultSize=0`.

## File does not exist on HDFS/LFS error from remote parfor

This error usually comes from incorrect HDFS configuration on the worker nodes. To investigate this, we recommend

- Testing if HDFS is accessible from the worker node: `hadoop fs -ls <file path>`
- Synchronize hadoop configuration across the worker nodes.
- Set the environment variable `HADOOP_CONF_DIR`. You may have to restart the cluster-manager to get the hadoop configuration. 

## JVM Garbage Collection related flags

We recommend providing 10% of maximum memory to young generation and using `-server` flag for robust garbage collection policy. 
For example: if you intend to use 20G driver and 60G executor, then please add following to your configuration:

	 spark-submit --driver-memory 20G --executor-memory 60G --conf "spark.executor.extraJavaOptions=-Xmn6G -server" --conf  "spark.driver.extraJavaOptions=-Xmn2G -server" ... 

## Memory overhead

Spark sets `spark.yarn.executor.memoryOverhead`, `spark.yarn.driver.memoryOverhead` and `spark.yarn.am.memoryOverhead` to be 10% of memory provided
to the executor, driver and YARN Application Master respectively (with minimum of 384 MB). For certain workloads, the user may have to increase this
overhead to 12-15% of the memory budget.

## Network timeout

To avoid false-positive errors due to network failures in case of compute-bound scripts, the user may have to increase the timeout `spark.network.timeout` (default: 120s).

## Advanced developer statistics

Few of our operators (for example: convolution-related operator) and GPU backend allows an expert user to get advanced statistics
by setting the configuration `systemml.stats.extraGPU` and `systemml.stats.extraDNN` in the file SystemML-config.xml. 

## Out-Of-Memory on executors

Out-Of-Memory on executors is often caused due to side-effects of lazy evaluation and in-memory input data of Spark for large-scale problems. 
Though we are constantly improving our optimizer to address this scenario, a quick hack to resolve this is reducing the number of cores allocated to the executor.
We would highly appreciate if you file a bug report on our [issue tracker](https://issues.apache.org/jira/browse/SYSTEMML) if and when you encounter OOM.

## Native BLAS errors

Please see [the user guide of native backend](http://apache.github.io/systemml/native-backend).
