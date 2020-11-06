# Import MKL
if [ -f ~/intel/bin/compilervars.sh ]; then
    . ~/intel/bin/compilervars.sh intel64
else
    . /opt/intel/bin/compilervars.sh intel64
fi

export LOG4JPROP='scripts/perftest/conf/log4j-off.properties'
export SYSDS_QUIET=1

LogName='scripts/perftest/results/MM.log'
mkdir -p 'scripts/perftest/results'
rm -f $LogName

perf stat -d -d -d -r 5 \
    systemds scripts/perftest/scripts/MM.dml \
    -config scripts/perftest/conf/std.xml \
    -stats \
    -args 5000 5000 5000 1.0 1.0 3 \
    >>$LogName 2>&1

perf stat -d -d -d -r 5 \
    systemds scripts/perftest/scripts/MM.dml \
    -config scripts/perftest/conf/mkl.xml \
    -stats \
    -args 5000 5000 5000 1.0 1.0 3 \
    >>$LogName 2>&1