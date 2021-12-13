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

package org.apache.sysds.runtime.compress.lib;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AMorphingMMColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;

public final class CLALibUtils {
	protected static final Log LOG = LogFactory.getLog(CLALibUtils.class.getName());

	/**
	 * Combine all column groups that are constant types, this include empty and const.
	 * 
	 * @param in A Compressed matrix.
	 */
	public static void combineConstColumns(CompressedMatrixBlock in) {
		// Combine Constant type column groups, both empty and const.
		List<AColGroup> e = new ArrayList<>();
		List<AColGroup> c = new ArrayList<>();
		List<AColGroup> o = new ArrayList<>();
		for(AColGroup g : in.getColGroups()) {
			if(g instanceof ColGroupEmpty)
				e.add(g);
			else if(g instanceof ColGroupConst)
				c.add(g);
			else
				o.add(g);
		}

		if(e.size() < 1 && c.size() < 1)
			return;

		if(e.size() == 1)
			o.add(e.get(0));
		else if(e.size() > 1)
			o.add(combineEmpty(e));

		if(c.size() == 1)
			o.add(c.get(0));
		else if(c.size() > 1)
			o.add(combineConst(c));

		in.allocateColGroupList(o);
	}

	/**
	 * Helper method to determine if the column groups contains SDC or Constant groups.
	 * 
	 * @param groups The ColumnGroups to analyze
	 * @return A Boolean saying there is SDC groups or Constant groups.
	 */
	protected static boolean shouldPreFilter(List<AColGroup> groups) {
		for(AColGroup g : groups)
			if(g instanceof AMorphingMMColGroup || g instanceof ColGroupConst)
				return true;
		return false;
	}

	/**
	 * Helper method to filter out SDC Groups and remove all constant groups, to reduce computation.
	 * 
	 * @param groups The Column Groups
	 * @param constV The Constant vector to add common values from SDC and all values from constant groups
	 * @return The Filtered list of Column groups containing no SDC Groups but only SDCZero groups.
	 */
	protected static List<AColGroup> filterGroups(List<AColGroup> groups, double[] constV) {
		if(constV == null)
			return groups;

		final List<AColGroup> filteredGroups = new ArrayList<>();
		for(AColGroup g : groups) {
			if(g instanceof AMorphingMMColGroup)
				filteredGroups.add(((AMorphingMMColGroup) g).extractCommon(constV));
			else if(g instanceof ColGroupEmpty)
				continue;
			else if(g instanceof ColGroupConst)
				((ColGroupConst) g).addToCommon(constV);
			else
				filteredGroups.add(g);
		}
		return returnGroupIfFiniteNumbers(groups, filteredGroups, constV);
	}

	private static List<AColGroup> returnGroupIfFiniteNumbers(List<AColGroup> groups, List<AColGroup> filteredGroups,
		double[] constV) {
		for(double v : constV)
			if(!Double.isFinite(v))
				throw new NotImplementedException("Not handling if the values are not finite: " + Arrays.toString(constV));
		return filteredGroups;
	}

	private static AColGroup combineEmpty(List<AColGroup> e) {
		return new ColGroupEmpty(combineColIndexes(e));
	}

	private static AColGroup combineConst(List<AColGroup> c) {
		int[] resCols = combineColIndexes(c);

		double[] values = new double[resCols.length];
		for(AColGroup g : c) {
			final ColGroupConst cg = (ColGroupConst) g;
			final int[] colIdx = cg.getColIndices();
			final double[] colVals = cg.getValues();
			for(int i = 0; i < colIdx.length; i++) {
				int outId = Arrays.binarySearch(resCols, colIdx[i]);
				values[outId] = colVals[i];
			}
		}
		return ColGroupFactory.genColGroupConst(resCols, values);
	}

	private static int[] combineColIndexes(List<AColGroup> gs) {
		int numCols = 0;
		for(AColGroup g : gs)
			numCols += g.getNumCols();

		int[] resCols = new int[numCols];

		int index = 0;
		for(AColGroup g : gs)
			for(int c : g.getColIndices())
				resCols[index++] = c;

		Arrays.sort(resCols);
		return resCols;
	}
}
