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
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import org.apache.spark.api.java.function.VoidFunction;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.codegen.CodegenUtils;
import org.apache.sysml.runtime.controlprogram.paramserv.PSWorker;
import org.apache.sysml.runtime.controlprogram.parfor.RemoteParForUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.ProgramConverter;

import scala.Tuple2;

public class SparkPSWorker extends PSWorker implements VoidFunction<Tuple2<Integer, Tuple2<MatrixBlock, MatrixBlock>>>, Serializable {

	private static final long serialVersionUID = -8674739573419648732L;

	private String _program;
	private HashMap<String, byte[]> _clsMap;

	protected SparkPSWorker() {
		// No-args constructor used for deserialization
	}

	public SparkPSWorker(String updFunc, Statement.PSFrequency freq, int epochs, long batchSize, String program, HashMap<String, byte[]> clsMap) {
		_updFunc = updFunc;
		_freq = freq;
		_epochs = epochs;
		_batchSize = batchSize;
		_program = program;
		_clsMap = clsMap;
	}

	@Override
	public void call(Tuple2<Integer, Tuple2<MatrixBlock, MatrixBlock>> input) throws Exception {
		configureWorker(input);
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
	}
}
