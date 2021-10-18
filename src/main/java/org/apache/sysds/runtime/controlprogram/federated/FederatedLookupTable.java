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

import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;
import java.util.Objects;

import org.apache.log4j.Logger;

public class FederatedLookupTable {
	public static final String NOHOST = "nohost";

	protected static Logger log = Logger.getLogger(FederatedLookupTable.class);

	private final Map<FedUniqueCoordID, ExecutionContextMap> _lookup_table;

	public FederatedLookupTable() {
		_lookup_table = new ConcurrentHashMap<>();
	}

	public ExecutionContextMap getECM(String host, long pid) {
		FedUniqueCoordID funCID = new FedUniqueCoordID(host, pid);
		ExecutionContextMap ecm = _lookup_table.computeIfAbsent(funCID,
			k -> new ExecutionContextMap());
		if(ecm == null) {
			log.error("Computing federated execution context map failed. "
				+ "No valid resolution for " + funCID.toString() + " found.");
			throw new FederatedWorkerHandlerException("Computing federated execution context map failed. "
				+ "No valid resolution for " + funCID.toString() + " found.");
		}
		return ecm;
	}

	public boolean containsFunCID(String host, long pid, long src_var_id) {
		FedUniqueCoordID funCID = new FedUniqueCoordID(host, pid);
		return _lookup_table.containsKey(funCID);
	}

	@Override
	public String toString() {
		return _lookup_table.toString();
	}


	private static class FedUniqueCoordID {
		private final String _host;
		private final long _pid;

		public FedUniqueCoordID(String host, long pid) {
			_host = host;
			_pid = pid;
		}

		@Override
		public final boolean equals(Object obj) {
			if(this == obj)
				return true;
			if(obj == null)
				return false;
			if(!(obj instanceof FedUniqueCoordID))
				return false;

			FedUniqueCoordID funCID = (FedUniqueCoordID) obj;

			return Objects.equals(_host, funCID._host)
				&& (_pid == funCID._pid);
		}

		@Override
		public int hashCode() {
			return Objects.hash(_host, _pid);
		}

		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder();
			sb.append(_pid);
			sb.append("@");
			sb.append(_host);
			return sb.toString();
		}
	}
}

