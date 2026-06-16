#!/bin/bash
#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

# A script to execute the tests inside the docker container.

cd /github/workspace/src/main/cpp
./build.sh
cd /github/workspace

export MAVEN_OPTS="-Xmx512m"

# Printed when Maven fails to download an artifact (transient repo/network
# error), unlike genuine compilation or test failures which fail fast.
transient_mvn_error="Could not transfer artifact"

log="/tmp/sysdstest.log"
compile_log="$(mktemp)"
# test-compile downloads all dependencies; retry once on a transient repo
# error so the test run below can resolve them from the local cache.
mvn -ntp -B test-compile 2>&1 | tee "$compile_log" | stdbuf -oL grep -E "BUILD|Total time:|---|Building SystemDS"
compile_status=${PIPESTATUS[0]}

# True only when test-compile failed because of a transient repository download.
compile_transient_failure=false
[ "$compile_status" -ne 0 ] && grep -qE "$transient_mvn_error" "$compile_log" && compile_transient_failure=true
rm -f "$compile_log"

if [ "$compile_transient_failure" = true ]; then
	echo "Transient Maven repository error; retrying test-compile in 15s..."
	sleep 15
	mvn -ntp -B test-compile 2>&1 | stdbuf -oL grep -E "BUILD|Total time:|---|Building SystemDS"
else
	echo "No transient Maven repository error detected; no retry needed."
fi

# --- Hung-fork diagnostics -------------------------------------------------
# Some fork JVMs finish their tests but never exit (the JVM stays alive while
# surefire waits, eventually hitting the job timeout). The thread leak is not a
# live non-daemon Java thread (verified locally), so the stall happens at JVM
# shutdown -- a blocking shutdown hook or a stuck native frame. Neither is
# visible after the fork is killed, so we snapshot the fork's stacks the moment
# it goes idle, before surefire force-kills it.
#
# Detection: the surefire fork (its command line contains "surefirebooter")
# burns CPU while running tests; if its cumulative CPU time stops advancing for
# a sustained window while the process is still alive, it is stalled. We then
# emit Java (jstack -l), native (jstack -m), and per-thread kernel wait-channel
# (/proc/<pid>/task/*/wchan) snapshots to the job log and to target artifacts.
dump_dir="/github/workspace/target/thread-dumps"
mkdir -p "$dump_dir"
jstack_bin="$JAVA_HOME/bin/jstack"

hung_fork_watchdog() {
	local poll=3 idle_limit=12          # dump after ~12s of zero CPU progress
	local last_pid="" last_cpu=-1 idle=0 dumps=0
	while true; do
		sleep "$poll"
		# Locate the current surefire fork via its command line. Use bash string
		# matching (not grep) so the scan never matches its own helper process.
		local pid="" p cl
		for p in /proc/[0-9]*; do
			[ -r "$p/cmdline" ] || continue
			cl=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)
			case "$cl" in
				*surefirebooter*) pid="${p#/proc/}"; break ;;
			esac
		done
		if [ -z "$pid" ]; then last_pid=""; last_cpu=-1; idle=0; continue; fi

		# Cumulative CPU (utime+stime, clock ticks) from /proc/<pid>/stat.
		local stat rest cpu
		stat=$(cat "/proc/$pid/stat" 2>/dev/null) || continue
		rest=${stat#*") "}
		# shellcheck disable=SC2086
		set -- $rest
		cpu=$(( ${12} + ${13} ))

		if [ "$pid" != "$last_pid" ]; then
			last_pid="$pid"; last_cpu="$cpu"; idle=0; dumps=0; continue
		fi
		if [ "$cpu" = "$last_cpu" ]; then
			idle=$(( idle + poll ))
		else
			idle=0; last_cpu="$cpu"
		fi

		if [ "$idle" -ge "$idle_limit" ] && [ "$dumps" -lt 3 ]; then
			local ts f
			ts=$(date +%H%M%S)
			f="$dump_dir/stall_${pid}_${ts}.txt"
			{
				echo "================ STALLED SUREFIRE FORK ================"
				echo "time=$(date +%H:%M:%S) pid=$pid cpu_ticks=$cpu idle>=${idle}s dump#$((dumps+1))"
				echo "--- cmdline ---"; tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null; echo
				echo "--- /proc/$pid/status (State/Threads) ---"
				grep -E '^(State|Threads):' "/proc/$pid/status" 2>/dev/null
				echo "--- per-thread kernel wait channel (tid comm wchan) ---"
				for t in /proc/$pid/task/*; do
					echo "  ${t##*/} $(cat "$t/comm" 2>/dev/null) $(cat "$t/wchan" 2>/dev/null)"
				done
				echo "--- jstack -l (live attach: Java threads + locks) ---"
				"$jstack_bin" -l "$pid" 2>&1
				echo "--- jstack -F -l (forced: works when the JVM is unresponsive) ---"
				"$jstack_bin" -F -l "$pid" 2>&1
				echo "--- jstack -m (mixed Java+native frames) ---"
				"$jstack_bin" -m "$pid" 2>&1
				echo "======================================================"
			} 2>&1 | tee -a "$f"
			dumps=$(( dumps + 1 ))
		fi
	done
}

hung_fork_watchdog &
watchdog_pid=$!

mvn -ntp -B test -D maven.test.skip=false -D automatedtestbase.outputbuffering=true -D test=$1 2>&1 \
	| stdbuf -oL grep -Ev "already exists in destination.|Using incubator" \
	| tee $log

kill "$watchdog_pid" 2>/dev/null


grep_args="SUCCESS"
grepvals="$( tail -n 100 $log | grep $grep_args)"

if [[ $grepvals == *"SUCCESS"* ]]; then
	# Merge Federated test runs.
	# if merged jacoco exist temporarily rename to not overwrite.
	[ -f target/jacoco.exec ] && mv target/jacoco.exec target/jacoco_main.exec
	# merge jacoco files.
	mvn -ntp -B jacoco:merge 2>&1 | stdbuf -oL grep -E "jacoco"

	exit 0
else
	exit 1
fi
