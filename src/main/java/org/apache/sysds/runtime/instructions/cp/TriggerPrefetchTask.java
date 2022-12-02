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

package org.apache.sysds.runtime.instructions.cp;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.federated.FederatedStatistics;
import org.apache.sysds.runtime.lineage.LineageCache;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.utils.stats.SparkStatistics;

public class TriggerPrefetchTask implements Runnable {
	MatrixObject _prefetchMO;
	LineageItem _inputLi;

	public TriggerPrefetchTask(MatrixObject mo) {
		_prefetchMO = mo;
		_inputLi = null;
	}

	public TriggerPrefetchTask(MatrixObject mo, LineageItem li) {
		_prefetchMO = mo;
		_inputLi = li;
	}

	@Override
	public void run() {
		boolean prefetched = false;
		long t1 = System.nanoTime();
		synchronized (_prefetchMO) {
			// Having this check inside the critical section
			// safeguards against concurrent rmVar.
			if (_prefetchMO.isPendingRDDOps() || _prefetchMO.isFederated()) {
				// TODO: Add robust runtime constraints for federated prefetch
				// Execute and bring the result to local
				_prefetchMO.acquireReadAndRelease();
				prefetched = true;
			}
		}

		// Save the collected intermediate in the lineage cache
		if (_inputLi != null)
			LineageCache.putValueAsyncOp(_inputLi, _prefetchMO, prefetched, t1);

		if (DMLScript.STATISTICS && prefetched) {
			if (_prefetchMO.isFederated())
				FederatedStatistics.incAsyncPrefetchCount(1);
			else
				SparkStatistics.incAsyncPrefetchCount(1);
		}
	}

}
