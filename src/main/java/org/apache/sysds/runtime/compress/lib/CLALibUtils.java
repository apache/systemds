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
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AMorphingMMColGroup;
import org.apache.sysds.runtime.compress.colgroup.APreAgg;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.IFrameOfReferenceGroup;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.IIterate;

public final class CLALibUtils {
	protected static final Log LOG = LogFactory.getLog(CLALibUtils.class.getName());

	private CLALibUtils() {
		// private constructor
	}

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
			if(g instanceof AMorphingMMColGroup //
				|| g instanceof ColGroupConst //
				|| g instanceof ColGroupEmpty //
				|| g.isEmpty())
				return true;
		return false;
	}

	protected static boolean alreadyPreFiltered(List<AColGroup> groups, int nCol) {
		boolean constFound = false;
		for(AColGroup g : groups) {
			if(g instanceof AMorphingMMColGroup || g instanceof ColGroupEmpty || g.isEmpty() ||
				(constFound && g instanceof ColGroupConst))
				return false;
			else if(g instanceof ColGroupConst){
				if(g.getNumCols() != nCol)
					return false;
				
				constFound = true;
			}
		}

		return true;
	}

	protected static double[] filterGroupsAndSplitPreAggOneConst(List<AColGroup> groups, List<AColGroup> noPreAggGroups,
		List<APreAgg> preAggGroups) {
		double[] consts = null;
		for(AColGroup g : groups) {
			if(g instanceof ColGroupConst)
			  consts = ((ColGroupConst) g).getValues();
			else if(g instanceof APreAgg)
				preAggGroups.add((APreAgg) g);
			else
				noPreAggGroups.add(g);
		}

		return consts;
	}

	protected static double[] filterGroupsAndSplitPreAggOneConst(List<AColGroup> groups, List<AColGroup> out) {
		double[] consts = null;
		for(AColGroup g : groups) {
			if(g instanceof ColGroupConst)
				consts = ((ColGroupConst) g).getValues();
			else
				out.add(g);
		}
		return consts;
	}

	/**
	 * Helper method to determine if the column groups contains Morphing or Frame of reference groups.
	 * 
	 * @param groups The groups to analyze
	 * @return A Boolean saying there is morphing or FOR groups.
	 */
	protected static boolean shouldPreFilterMorphOrRef(List<AColGroup> groups) {
		for(AColGroup g : groups)
			if(g instanceof AMorphingMMColGroup || g instanceof IFrameOfReferenceGroup || g instanceof ColGroupConst)
				return true;
		return false;
	}

	/**
	 * Detect if the list of groups contains FOR.
	 * 
	 * @param groups the groups
	 * @return If it contains FOR.
	 */
	protected static boolean shouldFilterFOR(List<AColGroup> groups) {
		for(AColGroup g : groups)
			if(g instanceof IFrameOfReferenceGroup)
				return true;
		return false;
	}

	protected static List<AColGroup> filterFOR(List<AColGroup> groups, double[] constV) {
		if(constV == null)
			return groups;
		final List<AColGroup> filteredGroups = new ArrayList<>();
		for(AColGroup g : groups)
			if(g instanceof IFrameOfReferenceGroup)
				filteredGroups.add(((IFrameOfReferenceGroup) g).extractCommon(constV));
			else
				filteredGroups.add(g);
		return filteredGroups;
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
			if(g instanceof ColGroupEmpty || g.isEmpty())
				continue;
			else if(g instanceof IFrameOfReferenceGroup)
				filteredGroups.add(((IFrameOfReferenceGroup) g).extractCommon(constV));
			else if(g instanceof AMorphingMMColGroup)
				filteredGroups.add(((AMorphingMMColGroup) g).extractCommon(constV));
			else if(g instanceof ColGroupConst)
				((ColGroupConst) g).addToCommon(constV);
			else
				filteredGroups.add(g);
		}
		return filteredGroups;
	}

	protected static void filterGroupsAndSplitPreAgg(List<AColGroup> groups, double[] constV,
		List<AColGroup> noPreAggGroups, List<APreAgg> preAggGroups) {
		for(AColGroup g : groups) {
			if(g instanceof APreAgg)
				preAggGroups.add((APreAgg) g);
			else if(g instanceof AMorphingMMColGroup) {
				AColGroup ga = ((AMorphingMMColGroup) g).extractCommon(constV);
				if(ga instanceof APreAgg)
					preAggGroups.add((APreAgg) ga);
				else if(!(ga instanceof ColGroupEmpty))
					throw new DMLCompressionException("I did not think this was a problem");
			}
			else if(g instanceof ColGroupEmpty)
				continue;
			else if(g instanceof ColGroupConst)
				((ColGroupConst) g).addToCommon(constV);
			else
				noPreAggGroups.add(g);
		}
	}

	protected static void splitPreAgg(List<AColGroup> groups, List<AColGroup> noPreAggGroups,
		List<APreAgg> preAggGroups) {
		for(AColGroup g : groups) {
			if(g instanceof APreAgg)
				preAggGroups.add((APreAgg) g);
			else if(g instanceof ColGroupEmpty)
				continue;
			else if(g instanceof ColGroupConst)
				throw new NotImplementedException();
			else
				noPreAggGroups.add(g);
		}
	}

	private static AColGroup combineEmpty(List<AColGroup> e) {
		return new ColGroupEmpty(combineColIndexes(e));
	}

	private static AColGroup combineConst(List<AColGroup> c) {
		IColIndex resCols = combineColIndexes(c);
		double[] values = new double[resCols.size()];
		for(AColGroup g : c) {
			final ColGroupConst cg = (ColGroupConst) g;
			final IColIndex colIdx = cg.getColIndices();
			final double[] colVals = cg.getValues();
			for(int i = 0; i < colIdx.size(); i++) {
				// Find the index in the result columns to add the value into.
				int outId = resCols.findIndex(colIdx.get(i));
				values[outId] = colVals[i];
			}
		}
		return ColGroupConst.create(resCols, values);
	}

	private static IColIndex combineColIndexes(List<AColGroup> gs) {
		return ColIndexFactory.combine(gs);
	}

	protected static double[] getColSum(List<AColGroup> groups, int nCols, int nRows) {
		return AColGroup.colSum(groups, new double[nCols], nRows);
	}

	protected static void addEmptyColumn(List<AColGroup> colGroups, int nCols) {

		// early abort loop
		for(AColGroup g : colGroups)
			if(g.getColIndices().size() == nCols)
				return; // there is some group that covers everything anyway

		Set<Integer> emptyColumns = new HashSet<>(nCols);
		for(int i = 0; i < nCols; i++)
			emptyColumns.add(i);

		for(AColGroup g : colGroups) {
			IIterate it = g.getColIndices().iterator();
			while(it.hasNext())
				emptyColumns.remove(it.next());
		}

		if(emptyColumns.size() != 0) {
			int[] emptyColumnsFinal = emptyColumns.stream().mapToInt(Integer::intValue).toArray();
			colGroups.add(new ColGroupEmpty(ColIndexFactory.create(emptyColumnsFinal)));
		}
		else
			return;
	}
}
