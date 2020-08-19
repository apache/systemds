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

package org.apache.sysds.runtime.lineage;

import java.util.ArrayList;
import java.util.Map;

import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.ForProgramBlock;
import org.apache.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysds.runtime.controlprogram.IfProgramBlock;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.utils.Explain;

public class LineageDedupUtils {
	public static final String DEDUP_DELIM = "_";
	private static Lineage _tmpLineage = null;
	private static Lineage _mainLineage = null;
	private static ArrayList<Long> _numDistinctPaths = new ArrayList<>();
	private static long _maxNumPaths = 0;
	private static int _numPaths = 0;
	
	public static boolean isValidDedupBlock(ProgramBlock pb, boolean inLoop) {
		// Only the last level loop-body in nested loop structure is valid for deduplication
		boolean ret = true; //basic program block
		if (pb instanceof FunctionProgramBlock) {
			FunctionProgramBlock fsb = (FunctionProgramBlock)pb;
			for (ProgramBlock cpb : fsb.getChildBlocks())
				ret &= isValidDedupBlock(cpb, inLoop);
		}
		else if (pb instanceof WhileProgramBlock) {
			if( inLoop ) return false;
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			for (ProgramBlock cpb : wpb.getChildBlocks())
				ret &= isValidDedupBlock(cpb, true);
		}
		else if (pb instanceof IfProgramBlock) {
			IfProgramBlock ipb = (IfProgramBlock) pb;
			for (ProgramBlock cpb : ipb.getChildBlocksIfBody())
				ret &= isValidDedupBlock(cpb, inLoop);
			for (ProgramBlock cpb : ipb.getChildBlocksElseBody())
				ret &= isValidDedupBlock(cpb, inLoop);
		}
		else if (pb instanceof ForProgramBlock) { //incl parfor
			if( inLoop ) return false;
			ForProgramBlock fpb = (ForProgramBlock) pb;
			for (ProgramBlock cpb : fpb.getChildBlocks())
				ret &= isValidDedupBlock(cpb, true);
		}
		return ret;
	}
	
	public static LineageDedupBlock computeDedupBlock(ProgramBlock fpb, ExecutionContext ec) {
		LineageDedupBlock ldb = new LineageDedupBlock();
		ec.getLineage().setInitDedupBlock(ldb);
		ldb.traceProgramBlocks(fpb.getChildBlocks(), ec);
		ec.getLineage().setInitDedupBlock(null);
		return ldb;
	}
	
	public static LineageDedupBlock initializeDedupBlock(ProgramBlock fpb, ExecutionContext ec) {
		LineageDedupBlock ldb = new LineageDedupBlock();
		ec.getLineage().setInitDedupBlock(ldb);
		// create/reuse a lineage object to trace the loop iterations
		initLocalLineage(ec);
		// save the original lineage object
		_mainLineage = ec.getLineage();
		// count and save the number of distinct paths
		ldb.setNumPathsInPBs(fpb.getChildBlocks(), ec);
		ec.getLineage().setInitDedupBlock(null);
		return ldb;
	}
	
	public static void setNewDedupPatch(LineageDedupBlock ldb, ProgramBlock fpb, ExecutionContext ec) {
		// no need to trace anymore if all the paths are taken, 
		// instead reuse the stored maps for this and future interations
		// NOTE: this optimization saves redundant tracing, but that
		//       kills reuse opportunities
		if (ldb.isAllPathsTaken())
			return;

		// copy the input LineageItems of the loop-body
		initLocalLineage(ec);
		ArrayList<String> inputnames = fpb.getStatementBlock().getInputstoSB();
		LineageItem[] liinputs = LineageItemUtils.getLineageItemInputstoSB(inputnames, ec);
		// TODO: find the inputs from the ProgramBlock instead of StatementBlock
		String ph = LineageItemUtils.LPLACEHOLDER;
		for (int i=0; i<liinputs.length; i++) {
			// Wrap the inputs with order-preserving placeholders.
			// An alternative way would be to replace the non-literal leaves with 
			// placeholders after each iteration, but that requires a full DAG
			// traversal after each iteration.
			LineageItem phInput = new LineageItem(ph+String.valueOf(i), new LineageItem[] {liinputs[i]});
			_tmpLineage.set(inputnames.get(i), phInput);
		}
		// also copy the dedupblock to trace the taken path (bitset)
		_tmpLineage.setDedupBlock(ldb);
		// attach the lineage object to the execution context
		ec.setLineage(_tmpLineage);
	}
	
	public static void replaceLineage(ExecutionContext ec) {
		// replace the local lineage with the original one
		ec.setLineage(_mainLineage);
	}
	
	public static void setDedupMap(LineageDedupBlock ldb, long takenPath) {
		// if this iteration took a new path, store the corresponding map
		if (ldb.getMap(takenPath) == null)
			ldb.setMap(takenPath, _tmpLineage.getLineageMap());
	}
	
	private static void initLocalLineage(ExecutionContext ec) {
		_tmpLineage = _tmpLineage == null ? new Lineage() : _tmpLineage;
		_tmpLineage.clearLineageMap();
		_tmpLineage.clearDedupBlock();
	}

	public static String mergeExplainDedupBlocks(ExecutionContext ec) {
		Map<ProgramBlock, LineageDedupBlock> dedupBlocks = ec.getLineage().getDedupBlocks();
		StringBuilder sb = new StringBuilder();
		// Gather all the DAG roots of all the paths in all the loops.
		for (Map.Entry<ProgramBlock, LineageDedupBlock> dblock : dedupBlocks.entrySet()) {
			if (dblock.getValue() != null) {
				String forKey = dblock.getKey().getStatementBlock().getName();
				LineageDedupBlock dedup = dblock.getValue();
				for (Map.Entry<Long, LineageMap> patch : dedup.getPathMaps().entrySet()) {
					for (Map.Entry<String, LineageItem> root : patch.getValue().getTraces().entrySet()) {
						// Encode all the information in the headers that're
						// needed by the deserialization logic.
						sb.append("patch");
						sb.append(DEDUP_DELIM);
						sb.append(root.getKey());
						sb.append(DEDUP_DELIM);
						sb.append(forKey);
						sb.append(DEDUP_DELIM);
						sb.append(patch.getKey());
						sb.append("\n");
						sb.append(Explain.explain(root.getValue()));
						sb.append("\n");
						
					}
				}
			}
		}
		return sb.toString();
	}
	
	//------------------------------------------------------------------------------
	/* The below static functions help to compute the number of distinct paths
	 * in any program block, and are used for diagnostic purposes. These will
	 * be removed in future.
	 */
	
	public static long computeNumPaths(ProgramBlock fpb, ExecutionContext ec) {
		if (fpb == null || fpb.getChildBlocks() == null)
			return 0;
		_numDistinctPaths.clear();
		long n = numPathsInPBs(fpb.getChildBlocks(), ec);
		if (n > _maxNumPaths) {
			_maxNumPaths = n;
			System.out.println("\nmax no of paths : " + _maxNumPaths + "\n");
		}
		return n;
	}
	
	public static long numPathsInPBs (ArrayList<ProgramBlock> pbs, ExecutionContext ec) {
		if (_numDistinctPaths.size() == 0) 
			_numDistinctPaths.add(0L);
		for (ProgramBlock pb : pbs)
			numPathsInPB(pb, ec, _numDistinctPaths);
		return _numDistinctPaths.size();
	}
	
	private static void numPathsInPB(ProgramBlock pb, ExecutionContext ec, ArrayList<Long> paths) {
		if (pb instanceof IfProgramBlock)
			numPathsInIfPB((IfProgramBlock)pb, ec, paths);
		else if (pb instanceof BasicProgramBlock)
			return;
		else
			return;
	}
	
	private static void numPathsInIfPB(IfProgramBlock ipb, ExecutionContext ec, ArrayList<Long> paths) {
		ipb.setLineageDedupPathPos(_numPaths++);
		ArrayList<Long> rep = new ArrayList<>();
		int pathKey = 1 << (_numPaths-1);
		for (long p : paths) {
			long pathIndex = p | pathKey;
			rep.add(pathIndex);
		}
		_numDistinctPaths.addAll(rep);
		for (ProgramBlock pb : ipb.getChildBlocksIfBody())
			numPathsInPB(pb, ec, rep);
		for (ProgramBlock pb : ipb.getChildBlocksElseBody())
			numPathsInPB(pb, ec, paths);
	}
}
