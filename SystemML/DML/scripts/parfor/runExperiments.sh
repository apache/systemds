export CLASSPATH=
export CLASSPATH=$CLASSPATH:.
export CLASSPATH=$CLASSPATH:bin/.
export CLASSPATH=$CLASSPATH:lib/log4j-1.2.15.jar
export CLASSPATH=$CLASSPATH:lib/commons-logging-1.1.1.jar
export CLASSPATH=$CLASSPATH:lib/opencsv-1.8.jar

export HADOOP_OPTS=-Xmx1024m
java -Xmx2048M com.ibm.bi.dml.runtime.controlprogram.parfor.test.Testsuite 

#nohup ./runExperiments.sh