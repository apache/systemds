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

package org.apache.sysds.runtime.controlprogram.federated;

import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.zip.Adler32;
import java.util.zip.Checksum;

import org.apache.sysds.api.DMLException;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheDataOutput;
import org.apache.sysds.runtime.controlprogram.caching.LazyWriteBuffer;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDHandler;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.utils.Statistics;

public class FederatedRequest implements Serializable {
	private static final long serialVersionUID = 5946781306963870394L;

	// commands sent to and excuted by federated workers
	public enum RequestType {
		READ_VAR,  // create variable for local data, read on first access
		PUT_VAR,   // receive data from main and store to local variable
		GET_VAR,   // return local variable to main
		EXEC_INST, // execute arbitrary instruction over
		EXEC_UDF,  // execute arbitrary user-defined function
		CLEAR,     // clear all variables and execution contexts (i.e., rmvar ALL)
		NOOP,      // no operation (part of request sequence and ID carrying)
	}

	private RequestType _method;
	private long _id;
	private long _tid;
	private List<Object> _data;
	private boolean _checkPrivacy;
	private List<Long> _checksums;
	private long _pid;

	public FederatedRequest(RequestType method) {
		this(method, FederationUtils.getNextFedDataID(), new ArrayList<>());
	}

	public FederatedRequest(RequestType method, long id) {
		this(method, id, new ArrayList<>());
	}

	public FederatedRequest(RequestType method, long id, Object ... data) {
		this(method, id, Arrays.asList(data));
	}

	public FederatedRequest(RequestType method, long id, List<Object> data) {
		Statistics.incFederated(method);
		_method = method;
		_id = id;
		_data = data;
		_pid = Long.valueOf(IDHandler.obtainProcessID());
		setCheckPrivacy();
		if (DMLScript.LINEAGE && method == RequestType.PUT_VAR)
			setChecksum();
	}

	public RequestType getType() {
		return _method;
	}

	public long getID() {
		return _id;
	}

	public long getTID() {
		return _tid;
	}

	public void setTID(long tid) {
		_tid = tid;
	}

	public long getPID() {
		return _pid;
	}

	public Object getParam(int i) {
		return _data.get(i);
	}

	public FederatedRequest appendParam(Object obj) {
		_data.add(obj);
		return this;
	}

	public FederatedRequest appendParams(Object ... objs) {
		_data.addAll(Arrays.asList(objs));
		return this;
	}

	public int getNumParams() {
		return _data.size();
	}

	public FederatedRequest deepClone() {
		return new FederatedRequest(_method, _id, new ArrayList<>(_data));
	}

	public void setCheckPrivacy(boolean checkPrivacy){
		this._checkPrivacy = checkPrivacy;
	}

	public void setCheckPrivacy(){
		setCheckPrivacy(DMLScript.CHECK_PRIVACY);
	}

	public boolean checkPrivacy(){
		return _checkPrivacy;
	}

	public void setChecksum() {
		// Calculate Adler32 checksum. This is used as a leaf node of Lineage DAGs
		// in the workers, and helps to uniquely identify a node (tracing PUT)
		// TODO: append lineageitem hash if checksum is not enough
		_checksums = new ArrayList<>();
		try {
			calcChecksum();
		}
		catch (IOException e) {
			throw new DMLException(e);
		}
	}

	public long getChecksum(int i) {
		return _checksums.get(i);
	}

	private void calcChecksum() throws IOException {
		for (Object ob : _data) {
			if (!(ob instanceof CacheBlock) && !(ob instanceof ScalarObject))
				continue;

			Checksum checksum = new Adler32();
			if (ob instanceof ScalarObject) {
				byte bytes[] = ((ScalarObject)ob).getStringValue().getBytes();
				checksum.update(bytes, 0, bytes.length);
				_checksums.add(checksum.getValue());
			}

			if (ob instanceof CacheBlock) {
				try {
					CacheBlock cb = (CacheBlock)ob;
					long cbsize = LazyWriteBuffer.getCacheBlockSize(cb);
					DataOutput dout = new CacheDataOutput(new byte[(int)cbsize]);
					cb.write(dout);
					byte bytes[] = ((CacheDataOutput) dout).getBytes();
					checksum.update(bytes, 0, bytes.length);
					_checksums.add(checksum.getValue());
				}
				catch(Exception ex) {
					throw new IOException("Failed to serialize cache block.", ex);
				}
			}
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder("FederatedRequest[");
		sb.append(_method); sb.append(";");
		sb.append(_id); sb.append(";");
		sb.append("t"); sb.append(_tid); sb.append(";");
		if( _method != RequestType.PUT_VAR )
			sb.append(Arrays.toString(_data.toArray()));
		sb.append("]");
		return sb.toString();
	}
}
