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

package org.apache.sysml.runtime.controlprogram.paramserv.spark.rpc;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

import org.apache.sysml.runtime.instructions.cp.ListObject;

public class PSRpcResponse extends PSRpcObject {

	public static final int SUCCESS = 1;
	public static final int ERROR = 2;

	private int _status;
	private Object _data;	// Could be list object or exception

	public PSRpcResponse(ByteBuffer buffer) throws IOException {
		deserialize(buffer);
	}

	public PSRpcResponse(int status, Object data) {
		_status = status;
		_data = data;
	}

	public boolean isSuccessful() {
		return _status == SUCCESS;
	}

	public String getErrorMessage() {
		return (String) _data;
	}

	public ListObject getResultModel() {
		return (ListObject) _data;
	}

	@Override
	public void deserialize(ByteBuffer buffer) throws IOException {
		_status = buffer.getInt();
		switch (_status) {
			case SUCCESS:
				_data = parseListObject(buffer);
				break;
			case ERROR:
				int messageSize = buffer.getInt();
				byte[] messageBytes = new byte[messageSize];
				buffer.get(messageBytes);
				_data = new String(messageBytes, "ISO-8859-1");
				break;
		}
	}

	@Override
	public ByteBuffer serialize() throws IOException {
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		DataOutputStream dos = new DataOutputStream(bos);
		dos.writeInt(_status);
		switch (_status) {
			case SUCCESS:
				if (!_data.equals(EMPTY_DATA)) {
					dos.write(deepSerializeListObject((ListObject) _data));
				}
				break;
			case ERROR:
				dos.writeInt(_data.toString().length());
				dos.write(_data.toString().getBytes());
				break;
		}
		dos.flush();
		return ByteBuffer.wrap(bos.toByteArray());
	}
}
