export CLASSPATH=
export CLASSPATH=$CLASSPATH:.
export CLASSPATH=$CLASSPATH:bin/.
export CLASSPATH=$CLASSPATH:lib/log4j-1.2.15.jar
export CLASSPATH=$CLASSPATH:lib/commons-logging-1.1.1.jar
export CLASSPATH=$CLASSPATH:lib/opencsv-1.8.jar

export HADOOP_OPTS="-Xmx1024m -Xms1024m"

hadoop jar SystemML.jar -f Corr_DataGen.dml -args 10000000 100

rm nohup.out

nohup hadoop jar SystemMLrep1.jar -f Corr_opt.dml -args 10000000 100
nohup hadoop jar SystemMLrep1.jar -f Corr_opt.dml -args 10000000 100
nohup hadoop jar SystemMLrep1.jar -f Corr_opt.dml -args 10000000 100

nohup hadoop jar SystemMLrep2.jar -f Corr_opt.dml -args 10000000 100
nohup hadoop jar SystemMLrep2.jar -f Corr_opt.dml -args 10000000 100
nohup hadoop jar SystemMLrep2.jar -f Corr_opt.dml -args 10000000 100

nohup hadoop jar SystemMLrep3.jar -f Corr_opt.dml -args 10000000 100
nohup hadoop jar SystemMLrep3.jar -f Corr_opt.dml -args 10000000 100
nohup hadoop jar SystemMLrep3.jar -f Corr_opt.dml -args 10000000 100

nohup hadoop jar SystemMLrep4.jar -f Corr_opt.dml -args 10000000 100
nohup hadoop jar SystemMLrep4.jar -f Corr_opt.dml -args 10000000 100
nohup hadoop jar SystemMLrep4.jar -f Corr_opt.dml -args 10000000 100

nohup hadoop jar SystemMLrep5.jar -f Corr_opt.dml -args 10000000 100
nohup hadoop jar SystemMLrep5.jar -f Corr_opt.dml -args 10000000 100
nohup hadoop jar SystemMLrep5.jar -f Corr_opt.dml -args 10000000 100

#exec time +2s for parsing and compilation