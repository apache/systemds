Usage
-----
The machine learning algorithms described in SystemML_Algorithms_Reference.pdf can be invoked
from the hadoop command line using the described, algorithm-specific parameters. 

Generic command line arguments arguments are provided by the help command below.

   hadoop jar SystemML.jar -? or -help 


Recommended configurations
--------------------------
1) JVM Heap Sizes: 
We recommend an equal-sized JVM configuration for clients, mappers, and reducers. For the client
process this can be done via

   export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m" 
   
where Xmx specifies the maximum heap size, Xms the initial heap size, and Xmn is size of the young 
generation. For Xmn values of equal or less than 15% of the max heap size, we guarantee the memory budget.

For mapper or reducer JVM configurations, the following properties can be specified in mapred-site.xml,
where 'child' refers to both mapper and reducer. If map and reduce are specified individually, they take 
precedence over the generic property.

  <property>
    <name>mapreduce.child.java.opts</name> <!-- synonym: mapred.child.java.opts -->
    <value>-Xmx2048m -Xms2048m -Xmn256m</value>
  </property>
  <property>
    <name>mapreduce.map.java.opts</name> <!-- synonym: mapred.map.java.opts -->
    <value>-Xmx2048m -Xms2048m -Xmn256m</value>
  </property>
  <property>
    <name>mapreduce.reduce.java.opts</name> <!-- synonym: mapred.reduce.java.opts -->
    <value>-Xmx2048m -Xms2048m -Xmn256m</value>
  </property>
 

2) CP Memory Limitation:
There exist size limitations for in-memory matrices. Dense in-memory matrices are limited to 16GB 
independent of their dimension. Sparse in-memory matrices are limited to 2G rows and 2G columns 
but the overall matrix can be larger. These limitations do only apply to in-memory matrices but 
NOT in HDFS or involved in MR computations. Setting HADOOP_CLIENT_OPTS below those limitations 
prevents runtime errors.

3) Transparent Huge Pages (on Red Hat Enterprise Linux 6):
Hadoop workloads might show very high System CPU utilization if THP is enabled. In case of such 
behavior, we recommend to disable THP with
   
   echo never > /sys/kernel/mm/redhat_transparent_hugepage/enabled
   
4) JVM Reuse:
Performance benefits from JVM reuse because data sets that fit into the mapper memory budget are 
reused across tasks per slot. However, Hadoop 1.0.3 JVM Reuse is incompatible with security (when 
using the LinuxTaskController). The workaround is to use the DefaultTaskController. SystemML provides 
a configuration property in SystemML-config.xml to enable JVM reuse on a per job level without
changing the global cluster configuration.
   
   <jvmreuse>false</jvmreuse> 
   
5) Number of Reducers:
The number of reducers can have significant impact on performance. SystemML provides a configuration
property to set the default number of reducers per job without changing the global cluster configuration.
In general, we recommend a setting of twice the number of nodes. Smaller numbers create less intermediate
files, larger numbers increase the degree of parallelism for compute and parallel write. In
SystemML-config.xml, set:
   
   <!-- default number of reduce tasks per MR job, default: 2 x number of nodes -->
   <numreducers>12</numreducers> 

6) SystemML temporary directories:
SystemML uses temporary directories in two different locations: (1) on local file system for temping from 
the client process, and (2) on HDFS for intermediate results between different MR jobs and between MR jobs 
and in-memory operations. Locations of these directories can be configured in SystemML-config.xml with the
following properties:

   <!-- local fs tmp working directory-->
   <localtmpdir>/tmp/systemml</localtmpdir>

   <!-- hdfs tmp working directory--> 
   <scratch>scratch_space</scratch> 
 