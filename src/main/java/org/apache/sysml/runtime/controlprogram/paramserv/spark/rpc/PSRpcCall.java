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
import static org.apache.sysml.runtime.util.ProgramConverter.LEVELIN;
import static org.apache.sysml.runtime.util.ProgramConverter.LEVELOUT;
import static org.apache.sysml.runtime.util.ProgramConverter.NEWLINE;

import java.nio.ByteBuffer;
import java.util.StringTokenizer;

import org.apache.sysml.runtime.instructions.cp.ListObject;
import org.apache.sysml.runtime.util.ProgramConverter;

public class PSRpcCall extends PSRpcObject {

	private static final String PS_RPC_CALL_BEGIN = CDATA_BEGIN + "PSRPCCALL" + LEVELIN;
	private static final String PS_RPC_CALL_END = LEVELOUT + CDATA_END;

	private String _method;
	private int _workerID;
	private ListObject _data;

	public PSRpcCall(String method, int workerID, ListObject data) {
		_method = method;
		_workerID = workerID;
		_data = data;
	}

	public PSRpcCall(ByteBuffer buffer) {
		deserialize(buffer);
	}

	public void deserialize(ByteBuffer buffer) {
		String input = new String(buffer.array());
		//header elimination
		String tmpin = input.replaceAll(NEWLINE, ""); //normalization
		tmpin = tmpin.substring(PS_RPC_CALL_BEGIN.length(), tmpin.length() - PS_RPC_CALL_END.length()); //remove start/end
		StringTokenizer st = new StringTokenizer(tmpin, COMPONENTS_DELIM);

		_method = st.nextToken();
		_workerID = Integer.valueOf(st.nextToken());
		_data = (ListObject) ProgramConverter.parseDataObject(st.nextToken())[1];
	}

	public ByteBuffer serialize() {
		StringBuilder sb = new StringBuilder();
		sb.append(PS_RPC_CALL_BEGIN);
		sb.append(NEWLINE);
		sb.append(_method);
		sb.append(NEWLINE);
		sb.append(COMPONENTS_DELIM);
		sb.append(NEWLINE);
		sb.append(_workerID);
		sb.append(NEWLINE);
		sb.append(COMPONENTS_DELIM);
		sb.append(NEWLINE);
		sb.append(ProgramConverter.serializeDataObject(DATA_KEY, _data));
		sb.append(NEWLINE);
		sb.append(PS_RPC_CALL_END);
		return ByteBuffer.wrap(sb.toString().getBytes());
	}

	public String getMethod() {
		return _method;
	}

	public int getWorkerID() {
		return _workerID;
	}

	public ListObject getData() {
		return _data;
	}

}
