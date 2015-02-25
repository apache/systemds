export CLASSPATH=
export CLASSPATH=$CLASSPATH:.
export CLASSPATH=$CLASSPATH:bin/.
export CLASSPATH=$CLASSPATH:lib/log4j-1.2.15.jar
export CLASSPATH=$CLASSPATH:lib/commons-cli-1.2.jar
export CLASSPATH=$CLASSPATH:lib/commons-configuration-1.6.jar
export CLASSPATH=$CLASSPATH:lib/commons-httpclient-3.0.1.jar
export CLASSPATH=$CLASSPATH:lib/commons-io-2.1.jar
export CLASSPATH=$CLASSPATH:lib/commons-lang-2.4.jar
export CLASSPATH=$CLASSPATH:lib/commons-logging-1.1.1.jar
export CLASSPATH=$CLASSPATH:lib/commons-math-2.2.jar
export CLASSPATH=$CLASSPATH:lib/opencsv-1.8.jar
export CLASSPATH=$CLASSPATH:lib/hadoop-core-1.1.1.jar
export CLASSPATH=$CLASSPATH:lib/jackson-core-asl-1.8.8.jar
export CLASSPATH=$CLASSPATH:lib/jackson-mapper-asl-1.8.8.jar
export CLASSPATH=$CLASSPATH:lib/nimble.jar
export CLASSPATH=$CLASSPATH:lib/blas.jar
export CLASSPATH=$CLASSPATH:lib/f2jutil.jar
export CLASSPATH=$CLASSPATH:lib/lapack.jar
export CLASSPATH=$CLASSPATH:lib/lapack_simple.jar
export CLASSPATH=$CLASSPATH:lib/netlib-java-0.9.2.jar
export CLASSPATH=$CLASSPATH:lib/xerbla.jar



export HADOOP_OPTS="-Xmx1024m -Xms1024"
java -Xmx2048M com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool 

#nohup ./runExperiments.sh