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

import org.apache.sysds.runtime.controlprogram.ForProgramBlock;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;

import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

import static org.apache.sysds.utils.Explain.explain;

public class Lineage {
	private final LineageMap _map;
	private final Stack<LineageDedupBlock> _initDedupBlock = new Stack<>();
	private final Stack<LineageDedupBlock> _activeDedupBlock = new Stack<>();
	private final Map<ForProgramBlock, LineageDedupBlock> _dedupBlocks = new HashMap<>();
	
	public Lineage() {
		_map = new LineageMap();
	}
	
	public Lineage(Lineage that) {
		_map = new LineageMap(that._map);
	}
	
	public void trace(Instruction inst, ExecutionContext ec) {
		if (_activeDedupBlock.empty())
			_map.trace(inst, ec);
	}
	
	public void tracePath(int block, Long path) {
		LineageMap lm = _activeDedupBlock.peek().getMap(block, path);
		if (lm != null)
			_map.processDedupItem(lm, path);
	}
	
	public LineageItem getOrCreate(CPOperand variable) {
		return _initDedupBlock.empty() ?
			_map.getOrCreate(variable) :
			_initDedupBlock.peek().getActiveMap().getOrCreate(variable);
	}
	
	public boolean contains(CPOperand variable) {
		return _initDedupBlock.empty() ?
			_map.containsKey(variable.getName()) :
			_initDedupBlock.peek().getActiveMap().containsKey(variable.getName());
	}
	
	public LineageItem get(String varName) {
		return _map.get(varName);
	}
	
	public void set(String varName, LineageItem li) {
		_map.set(varName, li);
	}
	
	public void setLiteral(String varName, LineageItem li) {
		_map.setLiteral(varName, li);
	}
	
	public LineageItem get(CPOperand variable) {
		return _initDedupBlock.empty() ?
			_map.get(variable) :
			_initDedupBlock.peek().getActiveMap().get(variable);
	}
	
	public void pushInitDedupBlock(LineageDedupBlock ldb) {
		_initDedupBlock.push(ldb);
	}
	
	public LineageDedupBlock popInitDedupBlock() {
		return _initDedupBlock.pop();
	}
	
	public void computeDedupBlock(ForProgramBlock fpb, ExecutionContext ec) {
		if (!_dedupBlocks.containsKey(fpb))
			_dedupBlocks.put(fpb, LineageDedupUtils.computeDedupBlock(fpb, ec));
		_activeDedupBlock.push(_dedupBlocks.get(fpb));
	}
	
	public void clearDedupBlock() {
		_activeDedupBlock.pop();
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
	
	public static void resetInternalState() {
		LineageItem.resetIDSequence();
		LineageCache.resetCache();
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
