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

import java.io.Serializable;

import org.apache.spark.api.java.function.VoidFunction;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamServer;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

import scala.Tuple2;

public class SparkPSWorker implements VoidFunction<Tuple2<Integer, Tuple2<MatrixBlock, MatrixBlock>>>,
		Serializable {

	private static final long serialVersionUID = -8674739573419648732L;

	public SparkPSWorker() {
		// No-args constructor used for deserialization
	}

	public SparkPSWorker(String updFunc, Statement.PSFrequency freq, int epochs, long batchSize,
			MatrixObject valFeatures, MatrixObject valLabels, ExecutionContext ec, ParamServer ps) {
	}

	@Override
	public void call(Tuple2<Integer, Tuple2<MatrixBlock, MatrixBlock>> input) throws Exception {
		int workerID = input._1;
	}
}
