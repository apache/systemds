#!/bin/bash

# 检查是否至少提供了一个参数
if [ $# -lt 1 ]; then
  echo "Usage: $0 <DML_FILE> [OPTIONS]"
  exit 1
fi

# 获取第一个参数作为DML文件
DML_FILE=$1
shift  # 移除第一个参数

# 构建java命令
JAVA_CMD="java -Xmx8g -Xms8g -Xmn400m \
    -agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=5005 \
    -Dlog4j.configuration=file:$SYSTEMDS_ROOT/conf/log4j.properties \
    -jar $SYSTEMDS_ROOT/target/systemds-3.3.0-SNAPSHOT.jar \
    -config $SYSTEMDS_ROOT/conf/SystemDS-config-defaults.xml \
    -f $DML_FILE \
    -exec singlenode"

# 将剩余参数追加到java命令
while [ $# -gt 0 ]; do
  JAVA_CMD="$JAVA_CMD $1"
  shift
done

# 输出并执行java命令
echo "Executing: $JAVA_CMD"
$JAVA_CMD   