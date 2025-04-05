 #Client mode spark-submit script
export SPARK_HOME=/home/hadoop/spark-3.3.1-bin-hadoop3
export HADOOP_CONF_DIR=/home/hadoop/hadoop-3.3.1/etc/hadoop

$SPARK_HOME/bin/spark-submit \
     --master yarn \
     --deploy-mode client \
     --driver-memory 20g \
     --num-executors 6 \
     --conf spark.driver.extraJavaOptions="-Xms20g -Xmn2g -Dlog4j.configuration=file:/home/mboehm/perftest/log4j.properties " \
     --conf spark.ui.showConsoleProgress=true \
     --conf spark.executor.heartbeatInterval=100s \
     --conf spark.network.timeout=512s \
     --executor-memory 200g \
     --executor-cores 48 \
      SystemDS.jar "$@" 