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
	private static final Timing executionTimer = new Timing(false);
	private static final LongAdder executionTime = new LongAdder();
	private static final LongAdder numWorkers = new LongAdder();
	private static final LongAdder setupTime = new LongAdder();
	private static final LongAdder gradientComputeTime = new LongAdder();
	private static final LongAdder aggregationTime = new LongAdder();
	private static final LongAdder localModelUpdateTime = new LongAdder();
	private static final LongAdder modelBroadcastTime = new LongAdder();
	private static final LongAdder batchIndexTime = new LongAdder();
	private static final LongAdder rpcRequestTime = new LongAdder();
	private static final LongAdder validationTime = new LongAdder();
	// Federated parameter server specifics (time is in milli sec)
	private static final LongAdder fedDataPartitioningTime = new LongAdder();
	private static final LongAdder fedWorkerComputingTime = new LongAdder();
	private static final LongAdder fedGradientWeightingTime = new LongAdder();
	private static final LongAdder fedCommunicationTime = new LongAdder();
	private static final LongAdder fedNetworkTime = new LongAdder(); // measures exactly how long it takes netty to send & receive data
	// Homomorphic encryption specifics (time is in milli sec)
	private static final LongAdder heEncryption = new LongAdder(); // SEALClient::encrypt
	private static final LongAdder heAccumulation = new LongAdder(); // SEALServer::accumulateCiphertexts
	private static final LongAdder hePartialDecryption = new LongAdder(); // SEALClient::partiallyDecrypt
	private static final LongAdder heDecryption = new LongAdder(); // SEALServer::average

	private static final LongAdder fedAggregation = new LongAdder(); // SEALServer::average

	public static void incWorkerNumber() {
		numWorkers.increment();
	}

	public static void incWorkerNumber(long n) {
		numWorkers.add(n);
	}

	public static Timing getExecutionTimer() {
		return executionTimer;
	}

	public static double getExecutionTime() {
		return executionTime.doubleValue();
	}

	public static void accExecutionTime(long n) {
		executionTime.add(n);
	}

	public static void accSetupTime(long t) {
		setupTime.add(t);
	}

	public static void accGradientComputeTime(long t) {
		gradientComputeTime.add(t);
	}

	public static void accAggregationTime(long t) {
		aggregationTime.add(t);
	}

	public static void accLocalModelUpdateTime(long t) {
		localModelUpdateTime.add(t);
	}

	public static void accModelBroadcastTime(long t) {
		modelBroadcastTime.add(t);
	}

	public static void accBatchIndexingTime(long t) {
		batchIndexTime.add(t);
	}

	public static void accRpcRequestTime(long t) {
		rpcRequestTime.add(t);
	}

	public static double getValidationTime() {
		return validationTime.doubleValue();
	}

	public static void accValidationTime(long t) {
		validationTime.add(t);
	}

	public static long getFedDataPartitioningTime() {
		return fedDataPartitioningTime.longValue();
	}

	public static void accFedDataPartitioningTime(long t) {
		fedDataPartitioningTime.add(t);
	}

	public static void accFedWorkerComputing(long t) {
		fedWorkerComputingTime.add(t);
	}

	public static void accFedNetworkTime(long t) {
		fedNetworkTime.add(t);
	}

	public static void accFedAggregation(long t) {
		fedAggregation.add(t);
	}

	public static void accFedGradientWeightingTime(long t) {
		fedGradientWeightingTime.add(t);
	}

	public static void accFedCommunicationTime(long t) {
		fedCommunicationTime.add(t);
	}

	public static void accHEEncryptionTime(long t) {
		heEncryption.add(t);
	}

	public static void accHEAccumulation(long t) {
		heAccumulation.add(t);
	}

	public static void accHEPartialDecryptionTime(long t) {
		hePartialDecryption.add(t);
	}

	public static void accHEDecryptionTime(long t) {
		heDecryption.add(t);
	}

	public static void reset() {
		executionTime.reset();
		numWorkers.reset();
		setupTime.reset();
		gradientComputeTime.reset();
		aggregationTime.reset();
		localModelUpdateTime.reset();
		modelBroadcastTime.reset();
		batchIndexTime.reset();
		rpcRequestTime.reset();
		validationTime.reset();
		fedDataPartitioningTime.reset();
		fedWorkerComputingTime.reset();
		fedGradientWeightingTime.reset();
		fedCommunicationTime.reset();
		fedNetworkTime.reset();
		heEncryption.reset();
		heAccumulation.reset();
		hePartialDecryption.reset();
		heDecryption.reset();
		fedAggregation.reset();
	}

	public static String displayStatistics() {
		if (numWorkers.longValue() > 0) {
			StringBuilder sb = new StringBuilder();
			sb.append(String.format("Paramserv total execution time:\t%.3f secs.\n", executionTime.doubleValue() / 1000));
			sb.append(String.format("Paramserv total num workers:\t%d.\n", numWorkers.longValue()));
			sb.append(String.format("Paramserv setup time:\t\t%.3f secs.\n", setupTime.doubleValue() / 1000));

			if(fedDataPartitioningTime.longValue() > 0) { 	//if data partitioning happens this is the federated case
				sb.append(displayFedPSStatistics());
				sb.append(String.format("PS fed global model agg time:\t%.3f secs.\n", aggregationTime.doubleValue() / 1000));
			}
			else {
				sb.append(String.format("Paramserv grad compute time:\t%.3f secs.\n", gradientComputeTime.doubleValue() / 1000));
				sb.append(String.format("Paramserv model update time:\t%.3f/%.3f secs.\n",
					localModelUpdateTime.doubleValue() / 1000, aggregationTime.doubleValue() / 1000));
				sb.append(String.format("Paramserv model broadcast time:\t%.3f secs.\n", modelBroadcastTime.doubleValue() / 1000));
				sb.append(String.format("Paramserv batch slice time:\t%.3f secs.\n", batchIndexTime.doubleValue() / 1000));
				sb.append(String.format("Paramserv RPC request time:\t%.3f secs.\n", rpcRequestTime.doubleValue() / 1000));
			}
			sb.append(String.format("Paramserv valdiation time:\t%.3f secs.\n", validationTime.doubleValue() / 1000));
			return sb.toString();
		}
		return "";
	}

	private static String displayFedPSStatistics() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("PS fed data partitioning time:\t%.3f secs.\n", fedDataPartitioningTime.doubleValue() / 1000));
		sb.append(String.format("PS fed comm time (cum):\t\t%.3f secs.\n", fedCommunicationTime.doubleValue() / 1000));
		sb.append(String.format("PS fed worker comp time (cum):\t%.3f secs.\n", fedWorkerComputingTime.doubleValue() / 1000));
		sb.append(String.format("PS fed grad. weigh. time (cum):\t%.3f secs.\n", fedGradientWeightingTime.doubleValue() / 1000));
		return sb.toString();
	}

	public static String displayFloStatistics() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("PS fed network time (cum):\t%.3f secs.\n", fedNetworkTime.doubleValue() / 1000));
		sb.append(String.format("PS fed agg time:\t%.3f secs.\n", fedAggregation.doubleValue() / 1000));
		sb.append(String.format("Paramserv grad compute time:\t%.3f secs.\n", gradientComputeTime.doubleValue() / 1000));
		sb.append(String.format("HE PS encryption time:\t%.3f secs.\n", heEncryption.doubleValue() / 1000));
		sb.append(String.format("HE PS accumulation time:\t%.3f secs.\n", heAccumulation.doubleValue() / 1000));
		sb.append(String.format("HE PS partial decryption time:\t%.3f secs.\n", hePartialDecryption.doubleValue() / 1000));
		sb.append(String.format("HE PS decryption time:\t%.3f secs.\n", heDecryption.doubleValue() / 1000));
		return sb.toString();
	}
}
