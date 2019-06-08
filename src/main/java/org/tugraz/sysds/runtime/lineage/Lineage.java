/*
 * Copyright 2019 Graz University of Technology
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

package org.tugraz.sysds.runtime.lineage;

import org.tugraz.sysds.runtime.controlprogram.ForProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.instructions.Instruction;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;

import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

public class Lineage {
	private static final LineageMap _globalLineages = new LineageMap();
	private static final Stack<LineageDedupBlock> _initDedupBlock = new Stack<>();
	private static final Stack<LineageDedupBlock> _activeDedupBlock = new Stack<>();
	private static final Map<ForProgramBlock, LineageDedupBlock> _dedupBlocks = new HashMap<>();
	
	private Lineage() {
		
	}
	
	public static void trace(Instruction inst, ExecutionContext ec) {
		if (_activeDedupBlock.empty())
			_globalLineages.trace(inst, ec);
	}
	
	public static void tracePath(int block, Long path) {
		LineageMap lm = _activeDedupBlock.peek().getMap(block, path);
		if (lm != null)
			_globalLineages.processDedupItem(lm, path);
	}
	
	public static LineageItem getOrCreate(CPOperand variable) {
		return _initDedupBlock.empty() ?
			_globalLineages.getOrCreate(variable) :
			_initDedupBlock.peek().getActiveMap().getOrCreate(variable);
	}
	
	public static boolean contains(CPOperand variable) {
		return _initDedupBlock.empty() ?
			_globalLineages.containsKey(variable.getName()) :
			_initDedupBlock.peek().getActiveMap().containsKey(variable.getName());
	}
	
	public static LineageItem get(CPOperand variable) {
		return _initDedupBlock.empty() ?
			_globalLineages.get(variable) :
			_initDedupBlock.peek().getActiveMap().get(variable);
	}
	
	public static void pushInitDedupBlock(LineageDedupBlock ldb) {
		_initDedupBlock.push(ldb);
	}
	
	public static LineageDedupBlock popInitDedupBlock() {
		return _initDedupBlock.pop();
	}
	
	public static void computeDedupBlock(ForProgramBlock fpb, ExecutionContext ec) {
		if (!_dedupBlocks.containsKey(fpb))
			_dedupBlocks.put(fpb, LineageDedupUtils.computeDedupBlock(fpb, ec));
		_activeDedupBlock.push(_dedupBlocks.get(fpb));
	}
	
	public static void clearDedupBlock() {
		_activeDedupBlock.pop();
	}
	
	public static void resetLineageMaps() {
		_globalLineages.resetLineageMaps();
	}
}
