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

package org.apache.sysml.runtime.controlprogram.paramserv;

import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;

public abstract class DataPartitioner {

	protected static final Log LOG = LogFactory.getLog(DataPartitioner.class.getName());

	public abstract void doPartitioning(List<LocalPSWorker> workers, MatrixObject features, MatrixObject labels);

	protected void setPartitionedData(List<LocalPSWorker> workers, List<MatrixObject> pfs, List<MatrixObject> pls) {
		if (pfs.size() < workers.size()) {
			if (LOG.isWarnEnabled()) {
				LOG.warn(String.format("There is only %d batches of data but has %d workers. "
					+ "Hence, reset the number of workers with %d.", pfs.size(), workers.size(), pfs.size()));
			}
			workers = workers.subList(0, pfs.size());
		}
		for (int i = 0; i < workers.size(); i++) {
			workers.get(i).setFeatures(pfs.get(i));
			workers.get(i).setLabels(pls.get(i));
		}
	}
}
