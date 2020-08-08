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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;
import java.util.concurrent.Future;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class FederationMap
{
	private long _ID = -1;
	private final Map<FederatedRange, FederatedData> _fedMap;
	
	public FederationMap(Map<FederatedRange, FederatedData> fedMap) {
		this(-1, fedMap);
	}
	
	public FederationMap(long ID, Map<FederatedRange, FederatedData> fedMap) {
		_ID = ID;
		_fedMap = fedMap;
	}
	
	public long getID() {
		return _ID;
	}
	
	public boolean isInitialized() {
		return _ID >= 0;
	}
	
	public FederatedRequest broadcast(CacheableData<?> data) {
		//prepare single request for all federated data
		long id = FederationUtils.getNextFedDataID();
		CacheBlock cb = data.acquireReadAndRelease();
		return new FederatedRequest(RequestType.PUT_VAR, id, cb);
	}
	
	public FederatedRequest broadcast(ScalarObject scalar) {
		//prepare single request for all federated data
		long id = FederationUtils.getNextFedDataID();
		return new FederatedRequest(RequestType.PUT_VAR, id, scalar);
	}
	
	public FederatedRequest[] broadcastSliced(CacheableData<?> data, boolean transposed) {
		//prepare separate requests for different slices
		long id = FederationUtils.getNextFedDataID();
		CacheBlock cb = data.acquireReadAndRelease();
		List<FederatedRequest> ret = new ArrayList<>();
		for(Entry<FederatedRange, FederatedData> e : _fedMap.entrySet()) {
			int rl = transposed ? 0 : e.getKey().getBeginDimsInt()[0];
			int ru = transposed ? cb.getNumRows()-1 : e.getKey().getEndDimsInt()[0]-1;
			int cl = transposed ? e.getKey().getBeginDimsInt()[0] : 0;
			int cu = transposed ? e.getKey().getEndDimsInt()[0]-1 : cb.getNumColumns()-1;
			CacheBlock tmp = cb.slice(rl, ru, cl, cu, new MatrixBlock());
			ret.add(new FederatedRequest(RequestType.PUT_VAR, id, tmp));
		}
		return ret.toArray(new FederatedRequest[0]);
	}
	
	@SuppressWarnings("unchecked")
	public Future<FederatedResponse>[] execute(FederatedRequest... fr) {
		List<Future<FederatedResponse>> ret = new ArrayList<>();
		for(Entry<FederatedRange, FederatedData> e : _fedMap.entrySet())
			ret.add(e.getValue().executeFederatedOperation(fr));
		return ret.toArray(new Future[0]);
	}
	
	@SuppressWarnings("unchecked")
	public Future<FederatedResponse>[] execute(FederatedRequest[] frSlices, FederatedRequest... fr) {
		//executes step1[] - step 2 - ... step4 (only first step federated-data-specific)
		List<Future<FederatedResponse>> ret = new ArrayList<>(); 
		int pos = 0;
		for(Entry<FederatedRange, FederatedData> e : _fedMap.entrySet())
			ret.add(e.getValue().executeFederatedOperation(addAll(frSlices[pos++], fr)));
		return ret.toArray(new Future[0]);
	}
	
	public List<Pair<FederatedRange, Future<FederatedResponse>>> requestFederatedData() {
		if( !isInitialized() )
			throw new DMLRuntimeException("Federated matrix read only supported on initialized FederatedData");
		
		List<Pair<FederatedRange, Future<FederatedResponse>>> readResponses = new ArrayList<>();
		FederatedRequest request = new FederatedRequest(RequestType.GET_VAR, _ID);
		for(Map.Entry<FederatedRange, FederatedData> e : _fedMap.entrySet())
			readResponses.add(new ImmutablePair<>(e.getKey(), 
				e.getValue().executeFederatedOperation(request)));
		return readResponses;
	}
	
	public void cleanup(long... id) {
		FederatedRequest request = new FederatedRequest(RequestType.EXEC_INST, -1,
			VariableCPInstruction.prepareRemoveInstruction(id).toString());
		for(FederatedData fd : _fedMap.values())
			fd.executeFederatedOperation(request);
	}
	
	private static FederatedRequest[] addAll(FederatedRequest a, FederatedRequest[] b) {
		FederatedRequest[] ret = new FederatedRequest[b.length + 1];
		ret[0] = a; System.arraycopy(b, 0, ret, 1, b.length);
		return ret;
	}
	
	public FederationMap copyWithNewID() {
		return copyWithNewID(FederationUtils.getNextFedDataID());
	}
	
	public FederationMap copyWithNewID(long id) {
		Map<FederatedRange, FederatedData> map = new TreeMap<>();
		//TODO handling of file path, but no danger as never written
		for( Entry<FederatedRange, FederatedData> e : _fedMap.entrySet() )
			map.put(new FederatedRange(e.getKey()), new FederatedData(e.getValue(), id));
		return new FederationMap(id, map);
	}

	public FederationMap rbind(long offset, FederationMap that) {
		for( Entry<FederatedRange, FederatedData> e : that._fedMap.entrySet() ) {
			_fedMap.put(
				new FederatedRange(e.getKey()).shift(offset, 0),
				new FederatedData(e.getValue(), _ID));
		}
		return this;
	}
}
