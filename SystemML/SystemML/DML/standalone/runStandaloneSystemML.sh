#!/bin/bash

# error help print
printUsageExit()
{
cat << EOF
Usage: $0 <dml-filename> [arguments] [-help]
    -help     - Print this usage message and exit
    Script internally invokes 'java -Xmx4g -Xms4g -Xmn400m -jar StandaloneSystemML.jar -f <dml-filename> -exec singlenode -config=SystemML-config.xml [Optional-Arguments]'
EOF
  exit 1
}


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

# invoke the jar with options and arguments
java -Xmx4g -Xms4g -Xmn400m -jar jSystemML.jar -f $1 -exec singlenode -config=SystemML-config.xml $2

