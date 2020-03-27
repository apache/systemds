/*
 * Modifications Copyright 2020 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.runtime.compress.cocode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.tugraz.sysds.runtime.compress.cocode.PlanningCoCoder.GroupableColInfo;

/**
 * Column group partitioning with static distribution heuristic.
 * 
 */
public class ColumnGroupPartitionerStatic extends ColumnGroupPartitioner {
	private static final int MAX_COL_PER_GROUP = 20;

	@Override
	public List<int[]> partitionColumns(List<Integer> groupCols, HashMap<Integer, GroupableColInfo> groupColsInfo) {
		List<int[]> ret = new ArrayList<>();
		int numParts = (int) Math.ceil((double) groupCols.size() / MAX_COL_PER_GROUP);
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
