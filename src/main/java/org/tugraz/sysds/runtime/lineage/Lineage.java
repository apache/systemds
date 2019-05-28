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

public class Lineage {
	
	private static LineageMap _globalLineages = new LineageMap();
	private static LineageDedupBlock _initDedupBlock = null;
	private static LineageDedupBlock _activeDedupBlock = null;
	
	private Lineage() {
	}
	
	public static void trace(Instruction inst, ExecutionContext ec) {
		if (_activeDedupBlock == null)
			_globalLineages.trace(inst, ec);
	}
	
	public static void tracePath(Long path) {
		LineageMap lm = _activeDedupBlock.getMap(path);
		_globalLineages.processDedupItem(lm, path);
	}
	
	public static LineageItem getOrCreate(CPOperand variable) {
		if (_initDedupBlock != null)
			return _initDedupBlock.getActiveMap().getOrCreate(variable);
		else
			return _globalLineages.getOrCreate(variable);
	}
	
	public static boolean contains(CPOperand variable) {
		return _initDedupBlock != null ?
				_initDedupBlock.getActiveMap().containsKey(variable.getName()) :
				_globalLineages.containsKey(variable.getName());
	}
	
	public static LineageItem get(CPOperand variable) {
		return _initDedupBlock != null ?
				_initDedupBlock.getActiveMap().get(variable) :
				_globalLineages.get(variable);
	}
	
	public static void setInitDedupBlock(LineageDedupBlock ldb) {
		_initDedupBlock = ldb;
	}
	
	public static void computeDedupBlock(ForProgramBlock fpb, ExecutionContext ec) {
		_activeDedupBlock = LineageDedupUtils.computeDistinctPaths(fpb, ec);
		ec.getLineagePath().initLastBranch();
	}
	
	public static void clearDedupBlock(ExecutionContext ec) {
		_activeDedupBlock = null;
		ec.getLineagePath().removeLastBranch();
	}
}
