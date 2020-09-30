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

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Future;

import org.apache.log4j.Logger;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;

public class FederatedLocalData extends FederatedData {
	protected final static Logger log = Logger.getLogger(FederatedWorkerHandler.class);

	private static final ExecutionContextMap ecm = new ExecutionContextMap();
	private static final FederatedWorkerHandler fwh = new FederatedWorkerHandler(ecm);

	private final CacheableData<?> _data;

	public FederatedLocalData(long id, CacheableData<?> data) {
		super(data.getDataType(), null, data.getFileName());
		_data = data;
		synchronized(ecm) {
			ecm.get(-1).setVariable(Long.toString(id), _data);
		}
		setVarID(id);
	}

	@Override
	boolean equalAddress(FederatedData that) {
		return that.getClass().equals(this.getClass());
	}

	@Override
	public FederatedData copyWithNewID(long varID) {
		return new FederatedLocalData(varID, _data);
	}

	@Override
	public synchronized Future<FederatedResponse> executeFederatedOperation(FederatedRequest... request) {
		return CompletableFuture.completedFuture(fwh.createResponse(request));
	}
}
