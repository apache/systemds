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

import org.apache.sysds.runtime.controlprogram.caching.CacheDataOutput;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.util.ByteBufferDataInput;

public class PSRpcResponse extends PSRpcObject {
	public enum Type  {
		SUCCESS,
		SUCCESS_EMPTY,
		ERROR,
	}
	
	private Type _status;
	private Object _data; // Could be list object or exception

	public PSRpcResponse(ByteBuffer buffer) throws IOException {
		deserialize(buffer);
	}

	public PSRpcResponse(Type status) {
		this(status, null);
	}
	
	public PSRpcResponse(Type status, Object data) {
		_status = status;
		_data = data;
		if( _status == Type.SUCCESS && data == null )
			_status = Type.SUCCESS_EMPTY;
	}

	public boolean isSuccessful() {
		return _status != Type.ERROR;
	}

	public String getErrorMessage() {
		return (String) _data;
	}

	public ListObject getResultModel() {
		return (ListObject) _data;
	}

	@Override
	public void deserialize(ByteBuffer buffer) throws IOException {
		ByteBufferDataInput dis = new ByteBufferDataInput(buffer);
		_status = Type.values()[dis.readInt()];
		switch (_status) {
			case SUCCESS:
				_data = readAndDeserialize(dis);
				break;
			case SUCCESS_EMPTY:
				break;
			case ERROR:
				_data = dis.readUTF();
				break;
		}
	}

	@Override
	public ByteBuffer serialize() throws IOException {
		int len = 4 + (_status==Type.SUCCESS ? getExactSerializedSize((ListObject)_data) :
			_status==Type.SUCCESS_EMPTY ? 0 : IOUtilFunctions.getUTFSize((String)_data));
		CacheDataOutput dos = new CacheDataOutput(len);
		dos.writeInt(_status.ordinal());
		switch (_status) {
			case SUCCESS:
				serializeAndWriteListObject((ListObject) _data, dos);
				break;
			case SUCCESS_EMPTY:
				break;
			case ERROR:
				dos.writeUTF(_data.toString());
				break;
		}
		return ByteBuffer.wrap(dos.getBytes());
	}
}
