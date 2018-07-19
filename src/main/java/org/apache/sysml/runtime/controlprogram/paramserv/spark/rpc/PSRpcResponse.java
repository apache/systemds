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

import static org.apache.sysml.runtime.util.ProgramConverter.CDATA_BEGIN;
import static org.apache.sysml.runtime.util.ProgramConverter.CDATA_END;
import static org.apache.sysml.runtime.util.ProgramConverter.COMPONENTS_DELIM;
import static org.apache.sysml.runtime.util.ProgramConverter.EMPTY;
import static org.apache.sysml.runtime.util.ProgramConverter.LEVELIN;
import static org.apache.sysml.runtime.util.ProgramConverter.LEVELOUT;

import java.nio.ByteBuffer;
import java.util.StringTokenizer;

import org.apache.sysml.runtime.instructions.cp.ListObject;
import org.apache.sysml.runtime.util.ProgramConverter;

public class PSRpcResponse extends PSRpcObject {

	public static final int SUCCESS = 1;
	public static final int ERROR = 2;

	private static final String PS_RPC_RESPONSE_BEGIN = CDATA_BEGIN + "PSRPCRESPONSE" + LEVELIN;
	private static final String PS_RPC_RESPONSE_END = LEVELOUT + CDATA_END;

	private int _status;
	private Object _data;	// Could be list object or exception

	public PSRpcResponse(ByteBuffer buffer) {
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
	public void deserialize(ByteBuffer buffer) {
		String input = bufferToString(buffer);
		//header elimination
		input = input.substring(PS_RPC_RESPONSE_BEGIN.length(), input.length() - PS_RPC_RESPONSE_END.length()); //remove start/end
		StringTokenizer st = new StringTokenizer(input, COMPONENTS_DELIM);

		_status = Integer.valueOf(st.nextToken());
		String data = st.nextToken();
		switch (_status) {
			case SUCCESS:
				if (data.equals(EMPTY)) {
					_data = null;
				} else {
					_data = ProgramConverter.parseDataObject(data)[1];
				}
				break;
			case ERROR:
				_data = data;
				break;
		}
	}

	@Override
	public ByteBuffer serialize() {
		StringBuilder sb = new StringBuilder();
		sb.append(PS_RPC_RESPONSE_BEGIN);
		sb.append(_status);
		sb.append(COMPONENTS_DELIM);
		switch (_status) {
			case SUCCESS:
				if (_data.equals(EMPTY_DATA)) {
					sb.append(EMPTY);
				} else {
					flushListObject((ListObject) _data);
					sb.append(ProgramConverter.serializeDataObject(DATA_KEY, (ListObject) _data));
				}
				break;
			case ERROR:
				sb.append(_data.toString());
				break;
		}
		sb.append(PS_RPC_RESPONSE_END);
		return ByteBuffer.wrap(sb.toString().getBytes());
	}
}
