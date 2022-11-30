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

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.lops.Checkpoint;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.utils.stats.SparkStatistics;

public class TriggerCheckpointTask implements Runnable {
	MatrixObject _remoteOperationsRoot;

	public TriggerCheckpointTask(MatrixObject mo) {
		_remoteOperationsRoot = mo;
	}

	@Override
	public void run() {
		boolean triggered = false;
		synchronized (_remoteOperationsRoot) {
			// FIXME: Handle double execution
			if (_remoteOperationsRoot.isPendingRDDOps()) {
				JavaPairRDD<?, ?> rdd = _remoteOperationsRoot.getRDDHandle().getRDD();
				rdd.persist(Checkpoint.DEFAULT_STORAGE_LEVEL).count();
				_remoteOperationsRoot.getRDDHandle().setCheckpointRDD(true);
				triggered = true;
			}
		}

		if (DMLScript.STATISTICS && triggered)
			SparkStatistics.incAsyncTriggerCheckpointCount(1);
	}
}
