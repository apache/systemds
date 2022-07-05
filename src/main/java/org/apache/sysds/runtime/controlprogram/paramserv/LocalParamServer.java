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

import org.apache.sysds.parser.Statement;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.ListObject;

public class LocalParamServer extends ParamServer {

	public LocalParamServer() {
		super();
	}

	public static LocalParamServer create(ListObject model, String aggFunc, Statement.PSUpdateType updateType,
		Statement.PSFrequency freq, ExecutionContext ec, int workerNum, String valFunc, int numBatchesPerEpoch,
		MatrixObject valFeatures, MatrixObject valLabels, int nbatches, boolean modelAvg, int numBackupWorkers)
	{
		return new LocalParamServer(model, aggFunc, updateType, freq, ec,
			workerNum, valFunc, numBatchesPerEpoch, valFeatures, valLabels, nbatches, modelAvg, numBackupWorkers);
	}

	protected LocalParamServer(ListObject model, String aggFunc, Statement.PSUpdateType updateType,
		Statement.PSFrequency freq, ExecutionContext ec, int workerNum, String valFunc, int numBatchesPerEpoch,
		MatrixObject valFeatures, MatrixObject valLabels, int nbatches, boolean modelAvg, int numBackupWorkers)
	{
		super(model, aggFunc, updateType, freq, ec, workerNum, valFunc, numBatchesPerEpoch, valFeatures, valLabels,
			nbatches, modelAvg, numBackupWorkers);
	}

	@Override
	public void push(int workerID, ListObject gradients) {
		updateGlobalModel(workerID, gradients);
	}

	@Override
	public ListObject pull(int workerID) {
		ListObject model;
		try {
			model = _modelMap.get(workerID).take();
		} catch (InterruptedException e) {
			throw new DMLRuntimeException(e);
		}
		return model;
	}
}
