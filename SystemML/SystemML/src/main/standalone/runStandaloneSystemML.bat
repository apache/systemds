@ECHO OFF

IF "%~1" == ""  GOTO Err
IF "%~1" == "-help" GOTO Msg
IF "%~1" == "-h" GOTO Msg

setLocal EnableDelayedExpansion

SET HADOOP_HOME=%CD%/lib/hadoop

set CLASSPATH=./lib/*
echo !CLASSPATH!
 
java -Xmx4g -Xms4g -Xmn400m -cp %CLASSPATH% com.ibm.bi.dml.api.DMLScript -f %1 -exec singlenode -config=SystemML-config.xml %2
GOTO End

:Err
ECHO "Wrong Usage. Please provide DML filename to be executed."
GOTO Msg

:Msg
ECHO "Usage: runStandaloneSystemML.bat <dml-filename> [arguments] [-help]"
ECHO "Script internally invokes 'java -Xmx4g -Xms4g -Xmn400m -jar jSystemML.jar -f <dml-filename> -exec singlenode -config=SystemML-config.xml [Optional-Arguments]'"

:End
