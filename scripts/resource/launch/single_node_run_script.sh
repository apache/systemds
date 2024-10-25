#!/usr/bin/env bash

# exit in case of error or unbound var
set -euo pipefail

# get file directory to allow finding the file with the utils
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

source single_node.env
source "$SCRIPT_DIR/single_node_utils.sh"

echo "Executing the DML script in single-node mode..."

# parse the JVM configs
generate_jvm_configs

if [[ -n "$SYSTEMDS_ARGS" ]]; then
    IFS=',' read -r -a args_array <<< "$SYSTEMDS_ARGS"
    LAUNCH_ARGS="${args_array[*]}"
    echo "Using arguments: $LAUNCH_ARGS"
else
    LAUNCH_ARGS=""
fi

if [[ -n "$SYSTEMDS_NVARGS" ]]; then
    IFS=',' read -r -a nvargs_array <<< "$SYSTEMDS_NVARGS"
    LAUNCH_NVARGS="${nvargs_array[*]}"
    echo "Using key-value arguments: $LAUNCH_NVARGS"
else
    LAUNCH_NVARGS=""
fi

EXEC_COMMAND="java -Xmx${JVM_MAX_MEM}m -Xms${JVM_START_MEM}m -Xmn${JVM_YOUNG_GEN_MEM}m
                   -jar /systemds/target/SystemDS.jar
                   -f $SYSTEMDS_PROGRAM -exec singlenode -stats -explain
                   $( [ -n "$LAUNCH_ARGS" ] && echo "-args $LAUNCH_ARGS" )
                   $( [ -n "$LAUNCH_NVARGS" ] && echo "-nvargs $LAUNCH_NVARGS" )"

# load the command to pure string
CMD=$(echo $EXEC_COMMAND)
echo -e "\nLaunching the program: $CMD"

ssh  -i "$KEYPAIR_NAME".pem "ec2-user@$PUBLIC_DNS_NAME" \
    "nohup bash -c '
    echo \"Start time: \$(date +\"%Y-%m-%d %H:%M:%S\") (\$(date +%s))\";
    $CMD;
    echo \"Finish time: \$(date +\"%Y-%m-%d %H:%M:%S\") (\$(date +%s))\"
    gzip -c output.log > output.log.gz &&
    aws s3 cp output.log.gz $LOG_URI/output_$INSTANCE_ID.log.gz --content-type \"text/plain\" --content-encoding \"gzip\" &&
    gzip -c error.log > error.log.gz &&
    aws s3 cp error.log.gz $LOG_URI/error_$INSTANCE_ID.log.gz --content-type \"text/plain\" --content-encoding \"gzip\" &&
    { if [ \"$AUTO_TERMINATION\" = true ]; then sudo shutdown now; fi; }' >> output.log 2>> error.log &"

echo "... the program has been launched"

if [ $AUTO_TERMINATION != true ]; then
    echo -e "\nYou need to check for its completion and stop/terminate the instance manually (automatic termination disabled)"
    exit 0
fi

echo -e "\nWaiting for the instance being stopped upon program completion (automatic termination enabled)..."
aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "...the DML finished, the logs where written to $LOG_URI and the EC2 instance was stopped"

echo "The instance will be terminated directly now..."
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null

echo "... termination was successful!"