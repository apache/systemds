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

package org.apache.sysml.runtime.controlprogram.paramserv.spark;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.spark.api.java.function.VoidFunction;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.codegen.CodegenUtils;
import org.apache.sysml.runtime.controlprogram.paramserv.LocalPSWorker;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysml.runtime.controlprogram.paramserv.spark.rpc.PSRpcFactory;
import org.apache.sysml.runtime.controlprogram.parfor.RemoteParForUtils;
import org.apache.sysml.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.ProgramConverter;
import org.apache.sysml.utils.Statistics;

import scala.Tuple2;

public class SparkPSWorker extends LocalPSWorker implements VoidFunction<Tuple2<Integer, Tuple2<MatrixBlock, MatrixBlock>>> {

	private static final long serialVersionUID = -8674739573419648732L;

	private String _program;
	private HashMap<String, byte[]> _clsMap;
	private String _host; // host ip of driver
	private long _rpcTimeout; // rpc ask timeout
	private String _aggFunc;

	public SparkPSWorker(String updFunc, String aggFunc, Statement.PSFrequency freq, int epochs, long batchSize, String program, HashMap<String, byte[]> clsMap, String host, long rpcTimeout) {
		_updFunc = updFunc;
		_aggFunc = aggFunc;
		_freq = freq;
		_epochs = epochs;
		_batchSize = batchSize;
		_program = program;
		_clsMap = clsMap;
		_host = host;
		_rpcTimeout = rpcTimeout;
	}

	@Override
	public String getWorkerName() {
		return String.format("Spark worker_%d", _workerID);
	}

	@Override
	public void call(Tuple2<Integer, Tuple2<MatrixBlock, MatrixBlock>> input) throws Exception {
		Timing tSetup = DMLScript.STATISTICS ? new Timing(true) : null;
		configureWorker(input);
		if (DMLScript.STATISTICS)
			Statistics.accPSSetupTime((long) tSetup.stop());
		call(); // Launch the worker
	}

	private void configureWorker(Tuple2<Integer, Tuple2<MatrixBlock, MatrixBlock>> input) throws IOException {
		_workerID = input._1;

		// Initialize codegen class cache (before program parsing)
		for (Map.Entry<String, byte[]> e : _clsMap.entrySet()) {
			CodegenUtils.getClassSync(e.getKey(), e.getValue());
		}

		// Deserialize the body to initialize the execution context
		SparkPSBody body = ProgramConverter.parseSparkPSBody(_program, _workerID);
		_ec = body.getEc();

		// Initialize the buffer pool and register it in the jvm shutdown hook in order to be cleanuped at the end
		RemoteParForUtils.setupBufferPool(_workerID);

		// Create the ps proxy
		_ps = PSRpcFactory.createSparkPSProxy(_host, _rpcTimeout);

		// Initialize the update function
		setupUpdateFunction(_updFunc, _ec);

		// Initialize the agg function
		_ps.setupAggFunc(_ec, _aggFunc);

		// Lazy initialize the matrix of features and labels
		setFeatures(ParamservUtils.newMatrixObject(input._2._1));
		setLabels(ParamservUtils.newMatrixObject(input._2._2));
		_features.enableCleanup(false);
		_labels.enableCleanup(false);
	}
}
