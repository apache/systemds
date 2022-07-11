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

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.util.LongAccumulator;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.runtime.codegen.CodegenUtils;
import org.apache.sysds.runtime.controlprogram.paramserv.rpc.PSRpcFactory;
import org.apache.sysds.runtime.controlprogram.parfor.RemoteParForUtils;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.ProgramConverter;

import scala.Tuple2;

public class SparkPSWorker extends LocalPSWorker implements VoidFunction<Tuple2<Integer, Tuple2<MatrixBlock, MatrixBlock>>> {

	private static final long serialVersionUID = -8674739573419648732L;

	private final String _program;
	private final boolean _isLocal;
	private final HashMap<String, byte[]> _clsMap;
	private final SparkConf _conf;
	private final int _port; // rpc port
	private final String _aggFunc;
	private final LongAccumulator _aSetup; // accumulator for setup time
	private final LongAccumulator _aWorker; // accumulator for worker number
	private final LongAccumulator _aUpdate; // accumulator for model update
	private final LongAccumulator _aIndex; // accumulator for batch indexing
	private final LongAccumulator _aGrad; // accumulator for gradients computing
	private final LongAccumulator _aRPC; // accumulator for rpc request
	private final LongAccumulator _nBatches; //number of executed batches
	private final LongAccumulator _nEpochs; //number of executed epoches

	public SparkPSWorker(String updFunc, String aggFunc, Statement.PSFrequency freq, int epochs, long batchSize, String program, boolean isLocal, HashMap<String, byte[]> clsMap, SparkConf conf, int port, LongAccumulator aSetup, LongAccumulator aWorker, LongAccumulator aUpdate, LongAccumulator aIndex, LongAccumulator aGrad, LongAccumulator aRPC, LongAccumulator aBatches, LongAccumulator aEpochs, int nbatches, boolean modelAvg) {
		_updFunc = updFunc;
		_aggFunc = aggFunc;
		_freq = freq;
		_epochs = epochs;
		_batchSize = batchSize;
		_program = program;
		_isLocal = isLocal;
		_clsMap = clsMap;
		_conf = conf;
		_port = port;
		_aSetup = aSetup;
		_aWorker = aWorker;
		_aUpdate = aUpdate;
		_aIndex = aIndex;
		_aGrad = aGrad;
		_aRPC = aRPC;
		_nBatches = aBatches;
		_nEpochs = aEpochs;
		_nbatches = nbatches;
		_modelAvg = modelAvg;
		
		// make SparkPSWorker serializable
		_tpool = null;
	}

	@Override
	public String getWorkerName() {
		return String.format("Spark worker_%d", _workerID);
	}

	@Override
	public void call(Tuple2<Integer, Tuple2<MatrixBlock, MatrixBlock>> input) throws Exception {
		Timing tSetup = new Timing(true);
		configureWorker(input);
		accSetupTime(tSetup);

		call(); // Launch the worker
	}

	private void configureWorker(Tuple2<Integer, Tuple2<MatrixBlock, MatrixBlock>> input)
			throws IOException, InterruptedException
	{
		_workerID = input._1;

		// Initialize codegen class cache (before program parsing)
		for (Map.Entry<String, byte[]> e : _clsMap.entrySet()) {
			CodegenUtils.getClassSync(e.getKey(), e.getValue());
		}

		// Deserialize the body to initialize the execution context
		SparkPSBody body = ProgramConverter.parseSparkPSBody(_program, _workerID);
		_ec = body.getEc();

		// Initialize the buffer pool and register it in the jvm shutdown hook in order to be cleanuped at the end
		RemoteParForUtils.setupBufferPool(_workerID, _isLocal);

		// Create the ps proxy
		_ps = PSRpcFactory.createSparkPSProxy(_conf, _port, _aRPC);

		// Initialize the update function
		setupUpdateFunction(_updFunc, _ec);

		// Initialize the agg function
		_ps.setupAggFunc(_ec, _aggFunc);

		// Lazy initialize the matrix of features and labels
		setFeatures(ParamservUtils.newMatrixObject(input._2._1, false));
		setLabels(ParamservUtils.newMatrixObject(input._2._2, false));
	}


	@Override
	protected void incWorkerNumber() {
		_aWorker.add(1);
	}

	@Override
	protected void accLocalModelUpdateTime(Timing time) {
		if( time != null )
			_aUpdate.add((long) time.stop());
	}

	@Override
	protected void accBatchIndexingTime(Timing time) {
		if( time != null )
			_aIndex.add((long) time.stop());
	}

	@Override
	protected void accGradientComputeTime(Timing time) {
		if( time != null )
			_aGrad.add((long) time.stop());
	}

	@Override
	protected void accNumEpochs(int n) {
		_nEpochs.add(n);
	}

	@Override
	protected void accNumBatches(int n) {
		_nBatches.add(n);
	}

	private void accSetupTime(Timing time) {
		if( time != null )
			_aSetup.add((long) time.stop());
	}
}
