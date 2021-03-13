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

package org.apache.sysds.runtime.compress;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.ColGroupType;

public class CompressionStatistics {

	private double lastPhase;
	public double ratio;
	public long originalSize;
	public long denseSize;
	public long estimatedSizeColGroups;
	public long estimatedSizeCols;
	public long size;

	private Map<ColGroupType, int[]> colGroupCounts;

	public CompressionStatistics() {
	}

	public void setNextTimePhase(double time) {
		lastPhase = time;
	}

	public double getLastTimePhase() {
		return lastPhase;
	}

	/**
	 * Set array of counts regarding col group types.
	 * 
	 * The position corresponds with the enum ordinal.
	 * 
	 * @param colGroups list of ColGroups used in compression.
	 */
	public void setColGroupsCounts(List<AColGroup> colGroups) {
		HashMap<ColGroupType, int[]> ret = new HashMap<>();
		for(AColGroup c : colGroups) {
			ColGroupType ct = c.getColGroupType();
			int colCount = c.getNumCols();
			int[] values;
			if(ret.containsKey(ct)) {
				values = ret.get(ct);
				values[0] += 1;
				values[1] += colCount;
			}
			else {
				values = new int[] {1, colCount};
			}
			ret.put(ct, values);
		}
		this.colGroupCounts = ret;
	}

	public Map<ColGroupType, int[]> getColGroups() {
		return colGroupCounts;
	}

	public String getGroupsTypesString() {
		StringBuilder sb = new StringBuilder();

		for(ColGroupType ctKey : colGroupCounts.keySet()) {
			sb.append(ctKey + ":" + colGroupCounts.get(ctKey)[0] + " ");
		}
		return sb.toString();
	}

	public String getGroupsSizesString() {
		StringBuilder sb = new StringBuilder();
		for(ColGroupType ctKey : colGroupCounts.keySet()) {

			sb.append(ctKey + ":" + colGroupCounts.get(ctKey)[1] + " ");
		}
		return sb.toString();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("Compression Statistics:\n");
		sb.append("\t" + getGroupsTypesString() + "\n");
		sb.append("\t" + getGroupsSizesString() + "\n");
		return sb.toString();
	}

}
