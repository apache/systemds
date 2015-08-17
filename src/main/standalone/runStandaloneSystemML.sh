#!/bin/bash

# error help print
printUsageExit()
{
cat << EOF
Usage: $0 <dml-filename> [arguments] [-help]
    -help     - Print this usage message and exit
EOF
  exit 1
}
#    Script internally invokes 'java -Xmx4g -Xms4g -Xmn400m -jar StandaloneSystemML.jar -f <dml-filename> -exec singlenode -config=SystemML-config.xml [Optional-Arguments]'

while getopts "h:" options; do
  case $options in
    h ) echo Warning: Help requested. Will exit after usage message;
        printUsageExit
        ;;
    \? ) echo Warning: Help requested. Will exit after usage message;
        printUsageExit
        ;;
    * ) echo Error: Unexpected error while processing options;
  esac
done

if [ -z $1 ] ; then
    echo "Wrong Usage.";
    printUsageExit;
fi

# Peel off first argument so that $@ contains arguments to DML script
SCRIPT_FILE=$1
shift

# Build up a classpath with all included libraries
CLASSPATH=""
for f in ./lib/*.jar; do
  CLASSPATH=${CLASSPATH}:$f;
done

# invoke the jar with options and arguments
java -Xmx4g -Xms4g -Xmn400m -cp ${CLASSPATH} com.ibm.bi.dml.api.DMLScript \
     -f ${SCRIPT_FILE} -exec singlenode -config=SystemML-config.xml \
     $@

