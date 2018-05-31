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

package org.apache.sysml.runtime.controlprogram.paramserv;

import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.ListObject;

public class LocalParamServer extends ParamServer {

	public LocalParamServer(ListObject model, String aggFunc, Statement.PSFrequency freq,
			Statement.PSUpdateType updateType, ExecutionContext ec, int workerNum, ListObject hyperParams) {
		super(model, aggFunc, freq, updateType, ec, workerNum, hyperParams);
	}

	@Override
	public void push(long workerID, ListObject gradients) {
		try {
			_gradientsQueue.put(new Gradient(workerID, gradients));
			launchService();
		} catch (InterruptedException e) {
			throw new DMLRuntimeException(
					String.format("Local param server: failed to push the gradients for worker_%d.", workerID), e);
		}
	}

	@Override
	public Data pull(long workerID) {
		ListObject model;
		try {
			model = _modelMap.get((int) workerID).take();
		} catch (InterruptedException e) {
			throw new DMLRuntimeException(
					String.format("Local param server: failed to pull the model for worker_%d", workerID), e);
		}
		setPulledState((int) workerID, true);
		return model;
	}
}
