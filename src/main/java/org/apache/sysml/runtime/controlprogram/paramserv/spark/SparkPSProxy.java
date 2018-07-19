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

import static org.apache.sysml.runtime.controlprogram.paramserv.spark.rpc.PSRpcObject.PULL;
import static org.apache.sysml.runtime.controlprogram.paramserv.spark.rpc.PSRpcObject.PUSH;

import org.apache.spark.network.client.TransportClient;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamServer;
import org.apache.sysml.runtime.controlprogram.paramserv.spark.rpc.PSRpcCall;
import org.apache.sysml.runtime.controlprogram.paramserv.spark.rpc.PSRpcResponse;
import org.apache.sysml.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysml.runtime.instructions.cp.ListObject;
import org.apache.sysml.utils.Statistics;

// TODO the rpc timeout should be able to be configured by user
public class SparkPSProxy extends ParamServer {

	private TransportClient _client;
	private static final long RPC_TIME_OUT = 1000 * 60 * 5;	// 5 minute of timeout

	public SparkPSProxy(TransportClient client) {
		super();
		_client = client;
	}

	@Override
	public void push(int workerID, ListObject value) {
		Timing tRpc = DMLScript.STATISTICS ? new Timing(true) : null;
		PSRpcResponse response = new PSRpcResponse(_client.sendRpcSync(new PSRpcCall(PUSH, workerID, value).serialize(), RPC_TIME_OUT));
		if (DMLScript.STATISTICS)
			Statistics.accPSRpcRequestTime((long) tRpc.stop());
		if (!response.isSuccessful()) {
			throw new DMLRuntimeException(String.format("SparkPSProxy: spark worker_%d failed to push gradients. \n%s", workerID, response.getErrorMessage()));
		}
	}

	@Override
	public ListObject pull(int workerID) {
		Timing tRpc = DMLScript.STATISTICS ? new Timing(true) : null;
		PSRpcResponse response = new PSRpcResponse(_client.sendRpcSync(new PSRpcCall(PULL, workerID, null).serialize(), RPC_TIME_OUT));
		if (DMLScript.STATISTICS)
			Statistics.accPSRpcRequestTime((long) tRpc.stop());
		if (!response.isSuccessful()) {
			throw new DMLRuntimeException(String.format("SparkPSProxy: spark worker_%d failed to pull models. \n%s", workerID, response.getErrorMessage()));
		}
		return response.getResultModel();
	}
}
