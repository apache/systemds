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

import static org.apache.sysds.runtime.controlprogram.paramserv.rpc.PSRpcObject.PULL;
import static org.apache.sysds.runtime.controlprogram.paramserv.rpc.PSRpcObject.PUSH;

import java.io.IOException;

import org.apache.spark.network.client.TransportClient;
import org.apache.spark.util.LongAccumulator;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.paramserv.rpc.PSRpcCall;
import org.apache.sysds.runtime.controlprogram.paramserv.rpc.PSRpcResponse;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.instructions.cp.ListObject;

public class SparkPSProxy extends ParamServer {

	private final TransportClient _client;
	private final long _rpcTimeout;
	private final LongAccumulator _aRPC;

	public SparkPSProxy(TransportClient client, long rpcTimeout, LongAccumulator aRPC) {
		super();
		_client = client;
		_rpcTimeout = rpcTimeout;
		_aRPC = aRPC;
	}

	private void accRpcRequestTime(Timing tRpc) {
		if (DMLScript.STATISTICS)
			_aRPC.add((long) tRpc.stop());
	}

	@Override
	public void push(int workerID, ListObject value) {
		Timing tRpc = DMLScript.STATISTICS ? new Timing(true) : null;
		PSRpcResponse response;
		try {
			response = new PSRpcResponse(_client.sendRpcSync(new PSRpcCall(PUSH, workerID, value).serialize(), _rpcTimeout));
		} catch (IOException e) {
			throw new DMLRuntimeException(String.format("SparkPSProxy: spark worker_%d failed to push gradients.", workerID), e);
		}
		accRpcRequestTime(tRpc);
		if (!response.isSuccessful()) {
			throw new DMLRuntimeException(String.format("SparkPSProxy: spark worker_%d failed to push gradients. \n%s", workerID, response.getErrorMessage()));
		}
	}

	@Override
	public ListObject pull(int workerID) {
		Timing tRpc = DMLScript.STATISTICS ? new Timing(true) : null;
		PSRpcResponse response;
		try {
			response = new PSRpcResponse(_client.sendRpcSync(new PSRpcCall(PULL, workerID, null).serialize(), _rpcTimeout));
		} catch (IOException e) {
			throw new DMLRuntimeException(String.format("SparkPSProxy: spark worker_%d failed to pull models.", workerID), e);
		}
		accRpcRequestTime(tRpc);
		if (!response.isSuccessful()) {
			throw new DMLRuntimeException(String.format("SparkPSProxy: spark worker_%d failed to pull models. \n%s", workerID, response.getErrorMessage()));
		}
		return response.getResultModel();
	}
}
