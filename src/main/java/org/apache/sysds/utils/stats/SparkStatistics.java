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
	private static long sparkCtxCreateTime = 0;
	private static final LongAdder sparkParallelize = new LongAdder();
	private static final LongAdder sparkParallelizeCount = new LongAdder();
	private static final LongAdder sparkCollect = new LongAdder();
	private static final LongAdder sparkCollectCount = new LongAdder();
	private static final LongAdder sparkBroadcast = new LongAdder();
	private static final LongAdder sparkBroadcastCount = new LongAdder();
	private static final LongAdder sparkAsyncPrefetchCount = new LongAdder();
	private static final LongAdder sparkAsyncBroadcastCount = new LongAdder();

	public static boolean createdSparkContext() {
		return sparkCtxCreateTime > 0;
	}

	public static void setSparkCtxCreateTime(long ns) {
		sparkCtxCreateTime = ns;
	}
	
	public static void accSparkParallelizeTime(long t) {
		sparkParallelize.add(t);
	}

	public static void incSparkParallelizeCount(long c) {
		sparkParallelizeCount.add(c);
	}

	public static void accSparkCollectTime(long t) {
		sparkCollect.add(t);
	}

	public static void incSparkCollectCount(long c) {
		sparkCollectCount.add(c);
	}

	public static void accSparkBroadCastTime(long t) {
		sparkBroadcast.add(t);
	}

	public static void incSparkBroadcastCount(long c) {
		sparkBroadcastCount.add(c);
	}

	public static void incSparkAsyncPrefetchCount(long c) {
		sparkAsyncPrefetchCount.add(c);
	}

	public static void incSparkAsyncBroadcastCount(long c) {
		sparkAsyncBroadcastCount.add(c);
	}

	public static long getSparkCollectCount() {
		return sparkCollectCount.longValue();
	}

	public static long getAsyncPrefetchCount() {
		return sparkAsyncPrefetchCount.longValue();
	}

	public static long getAsyncBroadcastCount() {
		return sparkAsyncBroadcastCount.longValue();
	}

	public static void reset() {
		sparkCtxCreateTime = 0;
		sparkBroadcast.reset();
		sparkBroadcastCount.reset();
		sparkAsyncPrefetchCount.reset();
		sparkAsyncBroadcastCount.reset();
	}

	public static String displaySparkStatistics() {
		StringBuilder sb = new StringBuilder();
		String lazy = SparkExecutionContext.isLazySparkContextCreation() ? "(lazy)" : "(eager)";
		sb.append("Spark ctx create time "+lazy+":\t"+
				String.format("%.3f", sparkCtxCreateTime*1e-9)  + " sec.\n" ); // nanoSec --> sec
		sb.append("Spark trans counts (par,bc,col):" +
				String.format("%d/%d/%d.\n", sparkParallelizeCount.longValue(),
						sparkBroadcastCount.longValue(), sparkCollectCount.longValue()));
		sb.append("Spark trans times (par,bc,col):\t" +
				String.format("%.3f/%.3f/%.3f secs.\n",
						sparkParallelize.longValue()*1e-9,
						sparkBroadcast.longValue()*1e-9,
						sparkCollect.longValue()*1e-9));
		if (OptimizerUtils.ASYNC_TRIGGER_RDD_OPERATIONS)
			sb.append("Spark async. count (pf,bc): \t" + 
					String.format("%d/%d.\n", getAsyncPrefetchCount(), getAsyncBroadcastCount()));
		return sb.toString();
	}
}
