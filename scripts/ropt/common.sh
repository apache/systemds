#!/bin/bash

# $1 key, $2 value
function set_config(){
    sed -i "" "s/\($1 *= *\).*/\1$2/" systemds_cluster.config
}

# $@ all arguments
# $REGION globally loaded var
execute() {
  local command=$@
  eval "$command --region $REGION &> /dev/null"
  if [ $? -ne 0 ]; then
    echo "Command '$command' failed. Exiting..."
    exit 1
  fi
}