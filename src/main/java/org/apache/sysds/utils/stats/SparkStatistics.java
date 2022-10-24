/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.utils.stats;

import java.util.concurrent.atomic.LongAdder;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;

public class SparkStatistics {
	private static long ctxCreateTime = 0;
	private static final LongAdder parallelizeTime = new LongAdder();
	private static final LongAdder parallelizeCount = new LongAdder();
	private static final LongAdder collectTime = new LongAdder();
	private static final LongAdder collectCount = new LongAdder();
	private static final LongAdder broadcastTime = new LongAdder();
	private static final LongAdder broadcastCount = new LongAdder();
	private static final LongAdder asyncPrefetchCount = new LongAdder();
	private static final LongAdder asyncBroadcastCount = new LongAdder();
	private static final LongAdder asyncTriggerRemoteCount = new LongAdder();

	public static boolean createdSparkContext() {
		return ctxCreateTime > 0;
	}

	public static void setCtxCreateTime(long ns) {
		ctxCreateTime = ns;
	}
	
	public static void accParallelizeTime(long t) {
		parallelizeTime.add(t);
	}

	public static void incParallelizeCount(long c) {
		parallelizeCount.add(c);
	}

	public static void accCollectTime(long t) {
		collectTime.add(t);
		incCollectCount(1);
	}

	private static void incCollectCount(long c) {
		collectCount.add(c);
	}

	public static void accBroadCastTime(long t) {
		broadcastTime.add(t);
	}

	public static void incBroadcastCount(long c) {
		broadcastCount.add(c);
	}

	public static void incAsyncPrefetchCount(long c) {
		asyncPrefetchCount.add(c);
	}

	public static void incAsyncBroadcastCount(long c) {
		asyncBroadcastCount.add(c);
	}

	public static void incAsyncTriggerRemoteCount(long c) {
		asyncTriggerRemoteCount.add(c);
	}

	public static long getSparkCollectCount() {
		return collectCount.longValue();
	}

	public static long getAsyncPrefetchCount() {
		return asyncPrefetchCount.longValue();
	}

	public static long getAsyncBroadcastCount() {
		return asyncBroadcastCount.longValue();
	}

	public static long getAsyncTriggerRemoteCount() {
		return asyncTriggerRemoteCount.longValue();
	}

	public static void reset() {
		ctxCreateTime = 0;
		parallelizeTime.reset();
		parallelizeCount.reset();
		broadcastTime.reset();
		broadcastCount.reset();
		collectTime.reset();
		collectCount.reset();
		asyncPrefetchCount.reset();
		asyncBroadcastCount.reset();
		asyncTriggerRemoteCount.reset();
	}

	public static String displayStatistics() {
		StringBuilder sb = new StringBuilder();
		String lazy = SparkExecutionContext.isLazySparkContextCreation() ? "(lazy)" : "(eager)";
		sb.append("Spark ctx create time "+lazy+":\t"+
				String.format("%.3f", ctxCreateTime*1e-9)  + " sec.\n" ); // nanoSec --> sec
		sb.append("Spark trans counts (par,bc,col):" +
				String.format("%d/%d/%d.\n", parallelizeCount.longValue(),
						broadcastCount.longValue(), collectCount.longValue()));
		sb.append("Spark trans times (par,bc,col):\t" +
				String.format("%.3f/%.3f/%.3f secs.\n",
						parallelizeTime.longValue()*1e-9,
						broadcastTime.longValue()*1e-9,
						collectTime.longValue()*1e-9));
		sb.append("Spark async. count (pf,bc,tr): \t" +
				String.format("%d/%d/%d.\n", getAsyncPrefetchCount(), getAsyncBroadcastCount(), getAsyncTriggerRemoteCount()));
		return sb.toString();
	}
}
