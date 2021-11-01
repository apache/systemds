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

import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;

public class ParamServStatistics {
	// Paramserv function stats (time is in milli sec)
	private static final Timing psExecutionTimer = new Timing(false);
	private static final LongAdder psExecutionTime = new LongAdder();
	private static final LongAdder psNumWorkers = new LongAdder();
	private static final LongAdder psSetupTime = new LongAdder();
	private static final LongAdder psGradientComputeTime = new LongAdder();
	private static final LongAdder psAggregationTime = new LongAdder();
	private static final LongAdder psLocalModelUpdateTime = new LongAdder();
	private static final LongAdder psModelBroadcastTime = new LongAdder();
	private static final LongAdder psBatchIndexTime = new LongAdder();
	private static final LongAdder psRpcRequestTime = new LongAdder();
	private static final LongAdder psValidationTime = new LongAdder();
	// Federated parameter server specifics (time is in milli sec)
	private static final LongAdder fedPSDataPartitioningTime = new LongAdder();
	private static final LongAdder fedPSWorkerComputingTime = new LongAdder();
	private static final LongAdder fedPSGradientWeightingTime = new LongAdder();
	private static final LongAdder fedPSCommunicationTime = new LongAdder();

	public static void incWorkerNumber() {
		psNumWorkers.increment();
	}

	public static void incWorkerNumber(long n) {
		psNumWorkers.add(n);
	}

	public static Timing getPSExecutionTimer() {
		return psExecutionTimer;
	}

	public static double getPSExecutionTime() {
		return psExecutionTime.doubleValue();
	}

	public static void accPSExecutionTime(long n) {
		psExecutionTime.add(n);
	}

	public static void accPSSetupTime(long t) {
		psSetupTime.add(t);
	}

	public static void accPSGradientComputeTime(long t) {
		psGradientComputeTime.add(t);
	}

	public static void accPSAggregationTime(long t) {
		psAggregationTime.add(t);
	}

	public static void accPSLocalModelUpdateTime(long t) {
		psLocalModelUpdateTime.add(t);
	}

	public static void accPSModelBroadcastTime(long t) {
		psModelBroadcastTime.add(t);
	}

	public static void accPSBatchIndexingTime(long t) {
		psBatchIndexTime.add(t);
	}

	public static void accPSRpcRequestTime(long t) {
		psRpcRequestTime.add(t);
	}

	public static double getPSValidationTime() {
		return psValidationTime.doubleValue();
	}

	public static void accPSValidationTime(long t) {
		psValidationTime.add(t);
	}

	public static long getFedPSDataPartitioningTime() {
		return fedPSDataPartitioningTime.longValue();
	}

	public static void accFedPSDataPartitioningTime(long t) {
		fedPSDataPartitioningTime.add(t);
	}

	public static void accFedPSWorkerComputing(long t) {
		fedPSWorkerComputingTime.add(t);
	}

	public static void accFedPSGradientWeightingTime(long t) {
		fedPSGradientWeightingTime.add(t);
	}

	public static void accFedPSCommunicationTime(long t) {
		fedPSCommunicationTime.add(t);
	}

	public static String displayParamServStatistics() {
		if (psNumWorkers.longValue() > 0) {
			StringBuilder sb = new StringBuilder();
			sb.append(String.format("Paramserv total execution time:\t%.3f secs.\n", psExecutionTime.doubleValue() / 1000));
			sb.append(String.format("Paramserv total num workers:\t%d.\n", psNumWorkers.longValue()));
			sb.append(String.format("Paramserv setup time:\t\t%.3f secs.\n", psSetupTime.doubleValue() / 1000));

			if(fedPSDataPartitioningTime.longValue() > 0) { 	//if data partitioning happens this is the federated case
				sb.append(displayFedPSStatistics());
				sb.append(String.format("PS fed global model agg time:\t%.3f secs.\n", psAggregationTime.doubleValue() / 1000));
			}
			else {
				sb.append(String.format("Paramserv grad compute time:\t%.3f secs.\n", psGradientComputeTime.doubleValue() / 1000));
				sb.append(String.format("Paramserv model update time:\t%.3f/%.3f secs.\n",
					psLocalModelUpdateTime.doubleValue() / 1000, psAggregationTime.doubleValue() / 1000));
				sb.append(String.format("Paramserv model broadcast time:\t%.3f secs.\n", psModelBroadcastTime.doubleValue() / 1000));
				sb.append(String.format("Paramserv batch slice time:\t%.3f secs.\n", psBatchIndexTime.doubleValue() / 1000));
				sb.append(String.format("Paramserv RPC request time:\t%.3f secs.\n", psRpcRequestTime.doubleValue() / 1000));
			}
			sb.append(String.format("Paramserv valdiation time:\t%.3f secs.\n", psValidationTime.doubleValue() / 1000));
			return sb.toString();
		}
		return "";
	}

	private static String displayFedPSStatistics() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("PS fed data partitioning time:\t%.3f secs.\n", fedPSDataPartitioningTime.doubleValue() / 1000));
		sb.append(String.format("PS fed comm time (cum):\t\t%.3f secs.\n", fedPSCommunicationTime.doubleValue() / 1000));
		sb.append(String.format("PS fed worker comp time (cum):\t%.3f secs.\n", fedPSWorkerComputingTime.doubleValue() / 1000));
		sb.append(String.format("PS fed grad. weigh. time (cum):\t%.3f secs.\n", fedPSGradientWeightingTime.doubleValue() / 1000));
		return sb.toString();
	}
}
