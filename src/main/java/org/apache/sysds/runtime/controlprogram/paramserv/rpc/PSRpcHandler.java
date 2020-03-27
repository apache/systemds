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

package org.apache.sysds.runtime.controlprogram.paramserv.rpc;

import static org.apache.sysds.runtime.controlprogram.paramserv.rpc.PSRpcCall.PULL;
import static org.apache.sysds.runtime.controlprogram.paramserv.rpc.PSRpcCall.PUSH;

import java.io.IOException;
import java.nio.ByteBuffer;

import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.spark.network.client.RpcResponseCallback;
import org.apache.spark.network.client.TransportClient;
import org.apache.spark.network.server.OneForOneStreamManager;
import org.apache.spark.network.server.RpcHandler;
import org.apache.spark.network.server.StreamManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.paramserv.LocalParamServer;
import org.apache.sysds.runtime.controlprogram.paramserv.rpc.PSRpcResponse.Type;
import org.apache.sysds.runtime.instructions.cp.ListObject;

public final class PSRpcHandler extends RpcHandler {

	private LocalParamServer _server;

	protected PSRpcHandler(LocalParamServer server) {
		_server = server;
	}

	@Override
	public void receive(TransportClient client, ByteBuffer buffer, RpcResponseCallback callback) {
		PSRpcCall call;
		try {
			call = new PSRpcCall(buffer);
		} catch (IOException e) {
			throw new DMLRuntimeException("PSRpcHandler: some error occurred when deserializing the rpc call.", e);
		}
		PSRpcResponse response = null;
		switch (call.getMethod()) {
			case PUSH:
				try {
					_server.push(call.getWorkerID(), call.getData());
					response = new PSRpcResponse(Type.SUCCESS_EMPTY);
				} catch (DMLRuntimeException exception) {
					response = new PSRpcResponse(Type.ERROR, ExceptionUtils.getFullStackTrace(exception));
				} finally {
					try {
						callback.onSuccess(response.serialize());
					} catch (IOException e) {
						throw new DMLRuntimeException("PSRpcHandler: some error occrred when wrapping the rpc response.", e);
					}
				}
				break;
			case PULL:
				ListObject data;
				try {
					data = _server.pull(call.getWorkerID());
					response = new PSRpcResponse(Type.SUCCESS, data);
				} catch (DMLRuntimeException exception) {
					response = new PSRpcResponse(Type.ERROR, ExceptionUtils.getFullStackTrace(exception));
				} finally {
					try {
						callback.onSuccess(response.serialize());
					} catch (IOException e) {
						throw new DMLRuntimeException("PSRpcHandler: some error occrred when wrapping the rpc response.", e);
					}
				}
				break;
			default:
				throw new DMLRuntimeException(String.format("Does not support the rpc call for method %s", call.getMethod()));
		}
	}

	@Override
	public StreamManager getStreamManager() {
		return new OneForOneStreamManager();
	}
}
