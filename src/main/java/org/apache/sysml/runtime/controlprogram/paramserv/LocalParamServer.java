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

import java.util.concurrent.ExecutionException;

import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.ListObject;

public class LocalParamServer extends ParamServer {

	public LocalParamServer(ListObject model, String aggFunc, Statement.PSUpdateType updateType, ExecutionContext ec,
			int workerNum) {
		super(model, aggFunc, updateType, ec, workerNum);
	}

	@Override
	public void push(int workerID, ListObject gradients) {
		try {
			_gradientsQueue.put(new Gradient(workerID, gradients));
		} catch (InterruptedException e) {
			throw new DMLRuntimeException(e);
		}
		try {
			launchService();
		} catch (ExecutionException | InterruptedException e) {
			throw new DMLRuntimeException("Aggregate service: some error occurred: ", e);
		}
	}

	@Override
	public Data pull(int workerID) {
		ListObject model;
		try {
			model = _modelMap.get(workerID).take();
		} catch (InterruptedException e) {
			throw new DMLRuntimeException(e);
		}
		return model;
	}
}
