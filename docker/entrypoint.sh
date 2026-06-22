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

target_dir="/github/workspace/target"
mkdir -p "$target_dir"
log="$target_dir/sysdstest.log"
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
# Outer guard: catch test-fork hangs that surefire's own timeouts miss, dump
# stacks for diagnosis, and kill the run before the job cap (kept just above the
# 600s per-fork timeout; MAX_RUNTIME is the absolute ceiling under the cap).
STALL_LIMIT="${SYSDS_TEST_STALL_LIMIT:-660}"
MAX_RUNTIME="${SYSDS_TEST_MAX_RUNTIME:-1600}"
dump_dir="$target_dir/thread-dumps"
mkdir -p "$dump_dir"
jstack_bin="${JAVA_HOME:+$JAVA_HOME/bin/}jstack"

# Emit the pid of a process and all of its descendants.
proc_tree() {
	local pid=$1 child
	for child in $(pgrep -P "$pid" 2>/dev/null); do proc_tree "$child"; done
	echo "$pid"
}

# SIGQUIT every JVM in the test tree (stacks relayed into $log) plus a jstack file.
dump_thread_stacks() {
	local reason="$1" root="$2" ts pid comm cmd jstack_file
	ts=$(date +%Y%m%d-%H%M%S)
	echo "================ HARD-GUARD THREAD DUMP: $reason ($ts) ================"
	for pid in $(proc_tree "$root"); do
		[ -r "/proc/$pid/comm" ] || continue
		comm=$(cat "/proc/$pid/comm" 2>/dev/null)
		case "$comm" in
			java|java.bin) ;;
			*) continue ;;
		esac
		cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-160)
		echo "---- SIGQUIT dump: pid=$pid comm=$comm cmd=$cmd ----"
		kill -3 "$pid" 2>/dev/null
		jstack_file="$dump_dir/jstack_${pid}_${ts}.txt"
		if timeout 30 "$jstack_bin" -l "$pid" > "$jstack_file" 2>&1; then
			echo "---- jstack dump: pid=$pid file=$jstack_file ----"
		else
			echo "---- jstack dump failed or timed out: pid=$pid file=$jstack_file ----"
		fi
		cat "$jstack_file" || true
		echo "---- end jstack dump: pid=$pid ----"
	done
	# Let the JVMs flush their dumps into the relayed output stream.
	sleep 12
	echo "================ END HARD-GUARD THREAD DUMP ($reason) ================"
}

# Background the run so the guard can watch it; $1 stays unquoted to keep the extra -D flags it carries.
( mvn -ntp -B test -D maven.test.skip=false -D automatedtestbase.outputbuffering=true -D test=$1 2>&1 \
	| stdbuf -oL grep -Ev "already exists in destination.|Using incubator" \
	| tee $log ) &
runner=$!

guard_tripped=false
start=$(date +%s)
prev_lines=-1
idle=0
interval=15
while kill -0 "$runner" 2>/dev/null; do
	sleep "$interval"
	now=$(date +%s)
	runtime=$((now - start))
	lines=$(wc -l < "$log" 2>/dev/null || echo 0)
	if [ "$lines" -eq "$prev_lines" ]; then
		idle=$((idle + interval))
	else
		idle=0
		prev_lines=$lines
	fi

	reason=""
	if [ "$idle" -ge "$STALL_LIMIT" ]; then
		reason="no test output for ${idle}s (stall limit ${STALL_LIMIT}s)"
	elif [ "$runtime" -ge "$MAX_RUNTIME" ]; then
		reason="exceeded absolute runtime ${runtime}s (max ${MAX_RUNTIME}s)"
	fi

	if [ -n "$reason" ]; then
		guard_tripped=true
		{
			echo ""
			echo "##[error] HARD GUARD TRIPPED: $reason"
			echo "Last test classes seen before the stall:"
			grep -E "Running org.apache" "$log" | tail -5
		} | tee -a "$log"
		dump_thread_stacks "$reason" "$runner" 2>&1 | tee -a "$log"
		for pid in $(proc_tree "$runner"); do kill -9 "$pid" 2>/dev/null; done
		break
	fi
done
wait "$runner" 2>/dev/null


grep_args="SUCCESS"
grepvals="$( tail -n 100 $log | grep $grep_args)"

if [ "$guard_tripped" = false ] && [[ $grepvals == *"SUCCESS"* ]]; then
	# Merge Federated test runs.
	# if merged jacoco exist temporarily rename to not overwrite.
	[ -f target/jacoco.exec ] && mv target/jacoco.exec target/jacoco_main.exec
	# merge jacoco files.
	mvn -ntp -B jacoco:merge 2>&1 | stdbuf -oL grep -E "jacoco"

	exit 0
else
	exit 1
fi
