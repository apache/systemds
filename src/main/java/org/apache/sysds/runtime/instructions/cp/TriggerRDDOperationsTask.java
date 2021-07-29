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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;

public class TriggerRDDOperationsTask implements Runnable {
	ExecutionContext ec;
	String _prefetchVar;

	public TriggerRDDOperationsTask(ExecutionContext ec, String var) {
		this.ec = ec;
		_prefetchVar = var;
	}

	@Override
	public void run() {
		MatrixObject mo = ec.getMatrixObject(_prefetchVar);
		//TODO: handle non-matrix objects
		if (!mo.isPendingRDDOps())
			throw new DMLRuntimeException("Variable to prefetch, " + _prefetchVar 
					+ " is either not empty or not a SPARK operation");

		synchronized (mo) {
				// Execute and bring those to local
				mo.acquireReadAndRelease();
		}
	}

}
