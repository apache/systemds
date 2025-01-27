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

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.ForProgramBlock;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static org.apache.sysds.utils.Explain.explain;

public class Lineage {
	//thread-/function-local lineage DAG
	private final LineageMap _map;
	
	//optional deduplication blocks (block := map of lineage patches per loop/function)
	//(for invalid loops, there is a null entry to avoid redundant validity checks)
	private final Map<ProgramBlock, LineageDedupBlock> _dedupBlocks = new HashMap<>();
	private LineageDedupBlock _activeDedupBlock = null; //used during dedup runtime
	private LineageDedupBlock _initDedupBlock = null;   //used during dedup init
	
	public Lineage() {
		_map = new LineageMap();
	}
	
	public Lineage(Lineage that) {
		_map = new LineageMap(that._map);
	}
	
	public void trace(Instruction inst, ExecutionContext ec) {
		if (inst.getOpcode().equalsIgnoreCase(Opcodes.TOSTRING.toString()))
			//Silently skip toString. TODO: trace toString
			return;

		if (_activeDedupBlock == null || !_activeDedupBlock.isAllPathsTaken() || !LineageCacheConfig.ReuseCacheType.isNone())
			_map.trace(inst, ec);
	}
	
	public void traceCurrentDedupPath(ProgramBlock pb, ExecutionContext ec) {
		if( _activeDedupBlock != null ) {
			ArrayList<String> inputnames = pb.getStatementBlock().getInputstoSB();
			LineageItem[] liinputs = LineageItemUtils.getLineageItemInputstoSB(inputnames, ec);
			long lpath = _activeDedupBlock.getPath();
			Map<String, Integer> dedupPatchHashList = LineageDedupUtils.setDedupMap(_activeDedupBlock, lpath);
			LineageMap lm = _activeDedupBlock.getMap(lpath);
			if (lm != null)
				_map.processDedupItem(lm, lpath, liinputs, pb.getStatementBlock().getName(), dedupPatchHashList);
		}
	}
	
	public LineageItem getOrCreate(CPOperand variable) {
		return _initDedupBlock == null ?
			_map.getOrCreate(variable) :
			_initDedupBlock.getActiveMap().getOrCreate(variable);
	}
	
	public boolean contains(CPOperand variable) {
		return _initDedupBlock == null ?
			_map.containsKey(variable.getName()) :
			_initDedupBlock.getActiveMap().containsKey(variable.getName());
	}
	
	public LineageItem get(String varName) {
		return _map.get(varName);
	}
	
	public void setDedupBlock(LineageDedupBlock ldb) {
		_activeDedupBlock = ldb;
	}
	
	public LineageMap getLineageMap() {
		return _map;
	}
	
	public Map<ProgramBlock, LineageDedupBlock> getDedupBlocks() {
		return _dedupBlocks;
	}
	
	public void set(String varName, LineageItem li) {
		_map.set(varName, li);
	}
	
	public void setLiteral(String varName, LineageItem li) {
		_map.setLiteral(varName, li);
	}
	
	public LineageItem get(CPOperand variable) {
		return _initDedupBlock == null ?
			_map.get(variable) :
			_initDedupBlock.getActiveMap().get(variable);
	}
	
	public void resetDedupPath() {
		if( _activeDedupBlock != null )
			_activeDedupBlock.resetPath();
	}
	
	public void setDedupPathBranch(int pos, boolean value) {
		if( _activeDedupBlock != null && value )
			_activeDedupBlock.setPathBranch(pos, value);
	}
	
	public void setInitDedupBlock(LineageDedupBlock ldb) {
		_initDedupBlock = ldb;
	}
	
	public void initializeDedupBlock(ProgramBlock pb, ExecutionContext ec) {
		if( !(pb instanceof ForProgramBlock || pb instanceof WhileProgramBlock) )
			throw new DMLRuntimeException("Invalid deduplication block: "+ pb.getClass().getSimpleName());
		LineageCacheConfig.setReuseLineageTraces(false);
		if (!_dedupBlocks.containsKey(pb)) {
			// valid only if doesn't contain a nested loop
			boolean valid = LineageDedupUtils.isValidDedupBlock(pb, false);
			// count distinct paths and store in the dedupblock
			_dedupBlocks.put(pb, valid? LineageDedupUtils.initializeDedupBlock(pb, ec) : null);
		}
		_activeDedupBlock = _dedupBlocks.get(pb); //null if invalid
	}
	
	public void createDedupPatch(ProgramBlock pb, ExecutionContext ec) {
		if (_activeDedupBlock != null)
			LineageDedupUtils.setNewDedupPatch(_activeDedupBlock, pb, ec);
	}
	
	public void clearDedupBlock() {
		_activeDedupBlock = null;
	}
	
	public void clearLineageMap() {
		_map.resetLineageMaps();
	}
	
	public Map<String,String> serialize() {
		Map<String,String> ret = new HashMap<>();
		for (Map.Entry<String,LineageItem> e : _map.getTraces().entrySet()) {
			ret.put(e.getKey(), explain(e.getValue()));
		}
		return ret;
	}
	
	public static Lineage deserialize(Map<String,String> serialLineage) {
		Lineage ret = new Lineage();
		for (Map.Entry<String,String> e : serialLineage.entrySet()) {
			ret.set(e.getKey(), LineageParser.parseLineageTrace(e.getValue()));
		}
		return ret;
	}

	public String serializeSingleTrace(CPOperand cpo) {
		LineageItem li = get(cpo);
		if(li == null)
			return null;
		return serializeSingleTrace(li);
	}

	public static String serializeSingleTrace(LineageItem linItem) {
		if(linItem == null)
			return null;

		return explain(linItem);
	}

	public static LineageItem deserializeSingleTrace(String serialLinTrace) {
		return LineageParser.parseLineageTrace(serialLinTrace);
	}
	
	public static void resetInternalState() {
		LineageItem.resetIDSequence();
		LineageCache.resetCache();
		LineageCacheStatistics.reset();
		LineageEstimator.resetEstimatorCache();
	}
	
	public static void setLinReusePartial() {
		LineageCacheConfig.setConfigTsmmCbind(ReuseCacheType.REUSE_PARTIAL);
	}

	public static void setLinReuseFull() {
		LineageCacheConfig.setConfigTsmmCbind(ReuseCacheType.REUSE_FULL);
	}
	
	public static void setLinReuseFullAndPartial() {
		LineageCacheConfig.setConfigTsmmCbind(ReuseCacheType.REUSE_HYBRID);
	}

	public static void setLinReuseNone() {
		LineageCacheConfig.setConfigTsmmCbind(ReuseCacheType.NONE);
	}
}
