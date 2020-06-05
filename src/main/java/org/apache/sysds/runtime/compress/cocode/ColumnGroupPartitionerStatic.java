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

package org.apache.sysds.runtime.compress.cocode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.cocode.PlanningCoCoder.GroupableColInfo;

/**
 * Column group partitioning with static distribution heuristic.
 */
public class ColumnGroupPartitionerStatic extends ColumnGroupPartitioner {

	@Override
	public List<int[]> partitionColumns(List<Integer> groupCols, HashMap<Integer, GroupableColInfo> groupColsInfo,
		CompressionSettings cs) {
		List<int[]> ret = new ArrayList<>();
		int numParts = (int) Math.ceil((double) groupCols.size() / cs.maxStaticColGroupCoCode);
		int partSize = (int) Math.ceil((double) groupCols.size() / numParts);
		for(int i = 0, pos = 0; i < numParts; i++, pos += partSize) {
			int[] tmp = new int[Math.min(partSize, groupCols.size() - pos)];
			for(int j = 0; j < partSize && pos + j < groupCols.size(); j++)
				tmp[j] = groupCols.get(pos + j);
			ret.add(tmp);
		}
		return ret;
	}
}
