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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroup.CompressionType;

public class CompressionStatistics {

	private ArrayList<Double> timePhases = new ArrayList<>();
	public double ratio;
	public long originalSize;
	public long estimatedSizeColGroups;
	public long estimatedSizeCols;
	public long size;

	private Map<CompressionType, int[]> colGroupCounts;

	public CompressionStatistics() {
	}

	public void setNextTimePhase(double time) {
		timePhases.add(time);
	}

	public double getLastTimePhase() {
		return timePhases.get(timePhases.size() - 1);
	}

	/**
	 * Set array of counts regarding col group types. 
	 * 
	 * The position corresponds with the enum ordinal.
	 * 
	 * @param colGroups list of ColGroups used in compression.
	 */
	public void setColGroupsCounts(List<ColGroup> colGroups) {
		HashMap<CompressionType, int[]> ret = new HashMap<>();
		for(ColGroup c : colGroups) {
			CompressionType ct = c.getCompType();
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

	public Map<CompressionType, int[]> getColGroups() {
		return colGroupCounts;
	}

	public ArrayList<Double> getTimeArrayList() {
		return timePhases;
	}

	public String getGroupsTypesString() {
		StringBuilder sb = new StringBuilder();

		for(CompressionType ctKey : colGroupCounts.keySet()) {
			sb.append(ctKey + ":" + colGroupCounts.get(ctKey)[0] + " ");
		}
		return sb.toString();
	}

	public String getGroupsSizesString() {
		StringBuilder sb = new StringBuilder();
		for(CompressionType ctKey : colGroupCounts.keySet()) {

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
