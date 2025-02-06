#!/usr/bin/env bash

mvn package > /dev/null
java -jar -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1008  100  100 1.0 1 30000
java -jar -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1008 1000  100 1.0 1 3000
java -jar -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1008 1000 1000 1.0 1 3000
java -jar -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1008  100  100 0.3 1 30000
java -jar -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1008 1000  100 0.3 1 3000
java -jar -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1008 1000 1000 0.3 1 3000

# java -jar -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1008  100  100 1.0 10 30000
# java -jar -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1008 1000  100 1.0 10 3000
# java -jar -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1008 1000 1000 1.0 10 1000
# java -jar -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1008  100  100 0.3 10 30000
# java -jar -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1008 1000  100 0.3 10 3000
# java -jar -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1008 1000 1000 0.3 10 1000

# java -jar -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1008  100  100 1.0 100 3000
# java -jar -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1008 1000  100 1.0 100 300
# java -jar -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1008 1000 1000 1.0 100 200
# java -jar -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1008  100  100 0.3 100 3000
# java -jar -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1008 1000  100 0.3 100 2000
# java -jar -XX:+UseNUMA target/systemds-3.3.0-SNAPSHOT-perf.jar 1008 1000 1000 0.3 100 1000