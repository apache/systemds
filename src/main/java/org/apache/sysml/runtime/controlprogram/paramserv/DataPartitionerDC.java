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

import java.util.ArrayList;
import java.util.List;

import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;

/**
 * Disjoint_Contiguous data partitioner:
 *
 * for each worker, use a right indexing
 * operation X[beg:end,] to obtain contiguous,
 * non-overlapping partitions of rows.
 */
public class DataPartitionerDC extends DataPartitioner {
	@Override
	public void doPartitioning(List<LocalPSWorker> workers, MatrixObject features, MatrixObject labels) {
		int workerNum = workers.size();
		List<MatrixObject> pfs = doPartitioning(workerNum, features);
		List<MatrixObject> pls = doPartitioning(workerNum, labels);
		setPartitionedData(workers, pfs, pls);
	}

	private List<MatrixObject> doPartitioning(int k, MatrixObject mo) {
		List<MatrixObject> list = new ArrayList<>();
		long stepSize = (long) Math.ceil(mo.getNumRows() / k);
		long begin = 1;
		while (begin < mo.getNumRows()) {
			long end = Math.min(begin - 1 + stepSize, mo.getNumRows());
			MatrixObject pmo = ParamservUtils.sliceMatrix(mo, begin, end);
			list.add(pmo);
			begin = end + 1;
		}
		return list;
	}
}
