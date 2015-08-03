HOW TO RUN THE PERFORMANCE SUITE
================================

Create a directory <...> on target machine on cluster, and copy from
repository SystemML/system-ml: machine/cluster:

   scripts/algorithms   to <...>/algorithms
   scripts/datagen      to <...>/datagen
   scripts/perftest     to <...>/perftest

Also copy:

   scripts/sparkDML     to <...>/perftest/                     // Edit sparkDML and set SPARK_HOME and SYSTEMML_HOME.
   target/SystemML.jar  to <...>/perftest/SystemML.jar
   test/config/SystemML-config.xml to <...>/perftest/SystemML-config.jar

   chmod +x <...>/perftest/*.sh sparkDML                       // Change permissions


Customize in runAll*.sh to choose data sizes as well as in gen*Data.sh.


Following alternative run modes are supported from <...>/perftest/

   ./runAll.sh $1 $2                // run all test 

   ./runAll.sh myperftest SPARK     // example

   $1 is used as a relative path in hdfs to store generated data,
   intermediate results, etc. $2 can be MR, SPARK, or ECHO. ECHO is
   meant for debugging the scripts as it just goes through all the
   scripts and outputs the invoked command line parameters.

   The scripts append to a trace/time file ./times.txt, and output log
   files in folder ./logs/*

   Below scripts can be invoked accordingly:

   ./runAllBinomial.sh $1 $2
   ./runAllClustering.sh $1 $2
   ./runAllMultinomial.sh $1 $2
   ./runAllRegression.sh $1 $2
   ./runAllStats.sh $1 $2

   ./genBinomialData.sh $1 $2
   ./genMultinomialData.sh $1 $2
   ./genClusteringData.sh $1 $2
   ./genDescriptiveStatisticsData.sh $1 $2
   ./genStratStatisticsData.sh $1 $2



      



