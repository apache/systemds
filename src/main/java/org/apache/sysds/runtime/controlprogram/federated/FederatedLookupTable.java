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

/**
 * Lookup table mapping from a FedUniqueCoordID (funCID) to an
 * ExecutionContextMap (ECM) so that every coordinator can address federated
 * variables with its own local sequential variable IDs. Therefore, the IDs
 * among different coordinators do not have to be distinct, as every
 * coordinator works with a seperate ECM at the FederatedWorker.
 */
public class FederatedLookupTable {
	// the NOHOST constant is needed for creating FederatedLocalData where there
	// is no actual network connection (and hence no host either)
	public static final String NOHOST = "nohost";

	protected static Logger log = Logger.getLogger(FederatedLookupTable.class);

	// stores the mapping between the funCID and the corresponding ExecutionContextMap
	private final Map<FedUniqueCoordID, ExecutionContextMap> _lookup_table;

	public FederatedLookupTable() {
		_lookup_table = new ConcurrentHashMap<>();
	}

	/**
	 * Get the ExecutionContextMap corresponding to the given host and pid of the
	 * requesting coordinator from the lookup table. Create a new
	 * ExecutionContextMap if there is no corresponding entry in the lookup table.
	 *
	 * @param host the host string of the requesting coordinator (usually IP address)
	 * @param pid the process id of the requesting coordinator
	 * @return ExecutionContextMap the ECM corresponding to the requesting coordinator
	 */
	public ExecutionContextMap getECM(String host, long pid) {
		log.trace("Getting the ExecutionContextMap for coordinator " + pid + "@" + host);
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

	/**
	 * Check if there is a mapped ExecutionContextMap for the coordinator
	 * with the given host and pid.
	 *
	 * @param host the host string of the requesting coordinator (usually IP address)
	 * @param pid the process id of the requesting coordinator
	 * @return boolean true if there is a lookup table entry, otherwise false
	 */
	public boolean containsFunCID(String host, long pid) {
		FedUniqueCoordID funCID = new FedUniqueCoordID(host, pid);
		return _lookup_table.containsKey(funCID);
	}

	@Override
	public String toString() {
		return _lookup_table.toString();
	}


	/**
	 * Class to collect the information needed to identify a specific coordinator.
	 */
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
