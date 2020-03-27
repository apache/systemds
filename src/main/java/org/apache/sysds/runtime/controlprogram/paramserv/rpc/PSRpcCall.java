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

import java.io.IOException;
import java.nio.ByteBuffer;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheDataOutput;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.util.ByteBufferDataInput;

public class PSRpcCall extends PSRpcObject {

	private int _method;
	private int _workerID;
	private ListObject _data;

	public PSRpcCall(int method, int workerID, ListObject data) {
		_method = method;
		_workerID = workerID;
		_data = data;
	}

	public PSRpcCall(ByteBuffer buffer) throws IOException {
		deserialize(buffer);
	}

	public int getMethod() {
		return _method;
	}

	public int getWorkerID() {
		return _workerID;
	}

	public ListObject getData() {
		return _data;
	}
	
	@Override
	public void deserialize(ByteBuffer buffer) throws IOException {
		ByteBufferDataInput dis = new ByteBufferDataInput(buffer);
		_method = dis.readInt();
		validateMethod(_method);
		_workerID = dis.readInt();
		if (dis.available() > 1)
			_data = readAndDeserialize(dis);
	}

	@Override
	public ByteBuffer serialize() throws IOException {
		int len = 8 + getExactSerializedSize(_data);
		CacheDataOutput dos = new CacheDataOutput(len);
		dos.writeInt(_method);
		dos.writeInt(_workerID);
		if (_data != null)
			serializeAndWriteListObject(_data, dos);
		return ByteBuffer.wrap(dos.getBytes());
	}
	
	private static void validateMethod(int method) {
		switch (method) {
			case PUSH:
			case PULL:
				break;
			default:
				throw new DMLRuntimeException("PSRpcCall: only support rpc method 'push' or 'pull'");
		}
	}
}
