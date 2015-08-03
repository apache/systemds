#!/bin/bash
#set -x

# Environment

SPARK_HOME=/home/biadmin/spark-1.4.0/spark-1.4.0-SNAPSHOT-bin-hadoop2.4
SYSTEMML_HOME="."

# Default Values

master=yarn-client
driver_memory=5G
num_executors=4
executor_memory=5G
executor_cores=12
conf="--conf spark.driver.maxResultSize=0 --conf spark.akka.frameSize=128"

# error help print

printUsageExit()
{
cat <<EOF

Usage: $0 [-h] [SPARK-SUBMIT OPTIONS] -f <dml-filename> [SYSTEMML OPTIONS]

   Examples:
      $0 -f genGNMF.dml --nvargs V=/tmp/V.mtx W=/tmp/W.mtx H=/tmp/H.mtx rows=100000 cols=800 k=50
      $0 --driver-memory 5G -f GNMF.dml --explain hops -nvargs ...
      $0 --master yarn-cluster -f hdfs:/user/GNMF.dml

   -h | -?  Print this usage message and exit

   SPARK-SUBMIT OPTIONS:
   --conf <property>=<value>   Configuration settings:                  
                                 spark.driver.maxResultSize            Default: 0 
                                 spark.akka.frameSize                  Default: 128
   --driver-memory <num>       Memory for driver (e.g. 512M)]          Default: 5G
   --master <string>           local | yarn-client | yarn-cluster]     Default: yarn-client
   --num-executors <num>       Number of executors to launch (e.g. 2)  Default: 4
   --executor-memory <num>     Memory per executor (e.g. 1G)           Default: 5G
   --executor-cores <num>      Memory per executor (e.g. )             Default: 12

   -f                          DML script file name, e.g. hdfs:/user/biadmin/test.dml

   SYSTEMML OPTIONS:
   --stats                     Monitor and report caching/recompilation statistics
   --explain [<string>]        Explain plan (hops, [runtime], recompile_hops, recompile_runtime)
   --nvargs <varName>=<value>  List of attributeName-attributeValue pairs
   --args <string>             List of positional argument values
EOF
  exit 1
}


# command line parameter processing

ARGS=`getopt -o h?f: --long conf:,driver-memory:,executor-cores:,executor-memory:,master:,num-executors:,nvargs:,args:,stats,explain: -- "$@"`
eval set -- "$ARGS"

while true ; do
  case "$1" in
    -h)                printUsageExit ; exit 1 ;;
    --master)          master=$2 ; shift 2 ;; 
    --driver-memory)   driver_memory=$2 ; shift 2 ;; 
    --num-executors)   num_executors=$2 ; shift 2 ;;
    --executor-memory) executor_memory=$2 ; shift 2 ;;
    --executor-cores)  executor_cores=$2 ; shift 2 ;;
    --conf)            conf=${conf}' --conf '$2 ; shift 2 ;;
     -f)               f=$2 ; shift 2 ;;
    --stats)           stats="-stats" ; shift 1 ;;
    --explain)         explain="-explain "$2 ; shift 2 ;;  
    --nvargs)          nvargs="-nvargs "$2 ; shift 2 ;;
    --args)            args="-args "$2 ; shift 2 ;;
    --)                shift ; if [ "$nvargs" != "" ]; then nvargs+=" $@" ; else args+=" $@" ; fi; shift ; break ;;
    *)                 echo "Error: Wrong usage. Try -h" ; exit 1 ;;
  esac
done


# SystemML Spark invocation

$SPARK_HOME/bin/spark-submit \
     --master ${master} \
     --driver-memory ${driver_memory} \
     --num-executors ${num_executors} \
     --executor-memory ${executor_memory} \
     --executor-cores ${executor_cores} \
     ${conf} \
     $SYSTEMML_HOME/SystemML.jar \
         -f ${f} \
         -config=$SYSTEMML_HOME/SystemML-config.xml \
         -exec spark \
         $explain \
         $stats \
         $nvargs $args
