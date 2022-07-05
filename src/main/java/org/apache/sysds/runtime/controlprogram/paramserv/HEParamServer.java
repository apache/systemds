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

package org.apache.sysds.runtime.controlprogram.paramserv;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.paramserv.homomorphicEncryption.PublicKey;
import org.apache.sysds.runtime.controlprogram.paramserv.homomorphicEncryption.SEALServer;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.instructions.cp.CiphertextMatrix;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.instructions.cp.PlaintextMatrix;
import org.apache.sysds.utils.stats.ParamServStatistics;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * This class implements Homomorphic Encryption (HE) for LocalParamServer. It only supports modelAvg=true.
 */
public class HEParamServer extends LocalParamServer {
	private int _thread_counter = 0;
	private final List<FederatedPSControlThread> _threads;
	private final List<Object> _result_buffer; // one per thread
	private Object _result;
	private final SEALServer _seal_server;

	public static HEParamServer create(ListObject model, String aggFunc, Statement.PSUpdateType updateType,
		Statement.PSFrequency freq, ExecutionContext ec, int workerNum, String valFunc, int numBatchesPerEpoch,
		MatrixObject valFeatures, MatrixObject valLabels, int nbatches, int numBackupWorkers)
	{
		NativeHEHelper.initialize();
		return new HEParamServer(model, aggFunc, updateType, freq, ec,
				workerNum, valFunc, numBatchesPerEpoch, valFeatures, valLabels, nbatches, numBackupWorkers);
	}

	private HEParamServer(ListObject model, String aggFunc, Statement.PSUpdateType updateType,
		Statement.PSFrequency freq, ExecutionContext ec, int workerNum, String valFunc,
		int numBatchesPerEpoch, MatrixObject valFeatures, MatrixObject valLabels, int nbatches, int numBackupWorkers)
	{
		super(model, aggFunc, updateType, freq, ec, workerNum, valFunc, numBatchesPerEpoch, valFeatures, valLabels,
			nbatches, true, numBackupWorkers);

		_seal_server = new SEALServer();

		_threads = Collections.synchronizedList(new ArrayList<>(workerNum));
		for (int i = 0; i < getNumWorkers(); i++) {
			_threads.add(null);
		}

		_result_buffer = new ArrayList<>(workerNum);
		resetResultBuffer();
	}

	public void registerThread(int thread_id, FederatedPSControlThread thread) {
		_threads.set(thread_id, thread);
	}

	private synchronized void resetResultBuffer() {
		_result_buffer.clear();
		for (int i = 0; i < getNumWorkers(); i++) {
			_result_buffer.add(null);
		}
	}

	public byte[] generateA() {
		return _seal_server.generateA();
	}

	public PublicKey aggregatePartialPublicKeys(PublicKey[] partial_public_keys) {
		return _seal_server.aggregatePartialPublicKeys(partial_public_keys);
	}

	/**
	 * this method collects all T Objects from each worker into a list and then calls f once on this list to produce
	 * another T, which it returns.
	 */
	@SuppressWarnings("unchecked")
	private synchronized <T,U> U collectAndDo(int workerId, T obj, Function<List<T>, U> f) {
		_result_buffer.set(workerId, obj);
		_thread_counter++;

		if (_thread_counter == getNumWorkers()) {
			List<T> buf = _result_buffer.stream().map(x -> (T)x).collect(Collectors.toList());
			_result = f.apply(buf);
			resetResultBuffer();
			_thread_counter = 0;
			notifyAll();
		} else {
			try {
				wait();
			} catch (InterruptedException i) {
				throw new RuntimeException("thread interrupted");
			}
		}

		return (U) _result;
	}

	private CiphertextMatrix[] homomorphicAggregation(List<ListObject> encrypted_models) {
		Timing tAgg = DMLScript.STATISTICS ? new Timing(true) : null;
		CiphertextMatrix[] result = new CiphertextMatrix[encrypted_models.get(0).getLength()];
		IntStream.range(0, encrypted_models.get(0).getLength()).forEach(matrix_idx -> {
			CiphertextMatrix[] summands = new CiphertextMatrix[encrypted_models.size()];
			for (int i = 0; i < encrypted_models.size(); i++) {
				summands[i] = (CiphertextMatrix) encrypted_models.get(i).getData(matrix_idx);
			}
			result[matrix_idx] = _seal_server.accumulateCiphertexts(summands);;
		});
		if (tAgg != null) {
			ParamServStatistics.accHEAccumulation((long)tAgg.stop());
		}
		return result;
	}

	private Void homomorphicAverage(CiphertextMatrix[] encrypted_sums, List<PlaintextMatrix[]> partial_decryptions) {
		Timing tDecrypt = DMLScript.STATISTICS ? new Timing(true) : null;

		MatrixObject[] result = new MatrixObject[partial_decryptions.get(0).length];

		IntStream.range(0, partial_decryptions.get(0).length).forEach(matrix_idx -> {
			PlaintextMatrix[] partial_plaintexts = new PlaintextMatrix[partial_decryptions.size()];
			for (int i = 0; i < partial_decryptions.size(); i++) {
				partial_plaintexts[i] = partial_decryptions.get(i)[matrix_idx];
			}

			result[matrix_idx] = _seal_server.average(encrypted_sums[matrix_idx], partial_plaintexts);
		});

		ListObject old_model = getResult();
		ListObject new_model = new ListObject(old_model);
		for (int i = 0; i < new_model.getLength(); i++) {
			new_model.set(i, result[i]);
		}

		if (tDecrypt != null) {
			ParamServStatistics.accHEDecryptionTime((long)tDecrypt.stop());
		}

		updateAndBroadcastModel(new_model, null);
		return null;
	}

	// this is only to be used in push()
	private Timing commTimer;
	private void startCommTimer() {
		commTimer = new Timing(true);
	}
	private long stopCommTimer() {
		return (long)commTimer.stop();
	}
	// ---------------------------------

	@Override
	public void push(int workerID, ListObject encrypted_model) {
		// wait for all updates and sum them homomorphically
		CiphertextMatrix[] homomorphic_sum = collectAndDo(workerID, encrypted_model, x -> {
			CiphertextMatrix[] res = this.homomorphicAggregation(x);
			this.startCommTimer();
			return res;
		});

		// get partial decryptions
		PlaintextMatrix[] partial_decryption = _threads.get(workerID).getPartialDecryption(homomorphic_sum);

		// do average and update global model
		collectAndDo(workerID, partial_decryption, x -> {
			ParamServStatistics.accFedNetworkTime(this.stopCommTimer());
			return this.homomorphicAverage(homomorphic_sum, x);
		});
	}
}
