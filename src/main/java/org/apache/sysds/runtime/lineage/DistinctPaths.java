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

import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.BasicProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.IfProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.ProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.instructions.Instruction;

import java.util.HashMap;
import java.util.Map;

public class DistinctPaths {
	private Map<Long, LineageMap> _distinctPaths = new HashMap<>();
	private Long _activePath = null;
	private int _branches = 0;
	
	public LineageMap getActiveMap() {
		if (_activePath == null || !_distinctPaths.containsKey(_activePath))
			throw new DMLRuntimeException("Active path in LineageDedupBlock could not be found.");
		return _distinctPaths.get(_activePath);
	}
	
	public LineageMap getMap(Long path) {
		if (!_distinctPaths.containsKey(path))
			throw new DMLRuntimeException("Given path in LineageDedupBlock could not be found.");
		return _distinctPaths.get(path);
	}
	
	public boolean empty(){
		return _distinctPaths.size() == 0;
	}
	
	public boolean pathExists(Long path) {
		return _distinctPaths.containsKey(path);
	}
	
	public void traceIfProgramBlock(IfProgramBlock ipb, ExecutionContext ec) {
		addPathsForBranch();
		traceElseBodyInstructions(ipb, ec);
		traceIfBodyInstructions(ipb, ec);
		_activePath = null;
	}
	
	public void traceBasicProgramBlock(BasicProgramBlock bpb, ExecutionContext ec) {
		if (_distinctPaths.size() == 0)
			_distinctPaths.put(0L, new LineageMap());
		for (Map.Entry<Long, LineageMap> entry : _distinctPaths.entrySet())
			traceInstructions(bpb, ec, entry);
		_activePath = null;
	}
	
	private void traceIfBodyInstructions(IfProgramBlock ipb, ExecutionContext ec) {
		// Add IfBody instructions to lower half of LineageMaps
		for (Map.Entry<Long, LineageMap> entry : _distinctPaths.entrySet())
			if (entry.getKey() >= _branches)
				for (ProgramBlock pb : ipb.getChildBlocksIfBody())
					traceInstructions(pb, ec, entry);
	}
	
	private void traceElseBodyInstructions(IfProgramBlock ipb, ExecutionContext ec) {
		// Add ElseBody instructions to upper half of LineageMaps
		for (Map.Entry<Long, LineageMap> entry : _distinctPaths.entrySet()) {
			if (entry.getKey() < _branches)
				for (ProgramBlock pb : ipb.getChildBlocksElseBody())
					traceInstructions(pb, ec, entry);
			else
				break;
		}
	}
	
	private void traceInstructions(ProgramBlock pb, ExecutionContext ec, Map.Entry<Long, LineageMap> entry) {
		if (!(pb instanceof BasicProgramBlock))
			throw new DMLRuntimeException("Only BasicProgramBlocks are allowed inside a LineageDedupBlock.");
		
		BasicProgramBlock bpb = (BasicProgramBlock) pb;
		for (Instruction inst : bpb.getInstructions()) {
			_activePath = entry.getKey();
			entry.getValue().trace(inst, ec);
		}
	}
	
	private void addPathsForBranch() {
		if (_distinctPaths.size() == 0) {
			_distinctPaths.put(0L, new LineageMap());
			_distinctPaths.put(1L, new LineageMap());
		} else {
			Map<Long, LineageMap> elseBranches = new HashMap<>();
			for (Map.Entry<Long, LineageMap> entry : _distinctPaths.entrySet()) {
				Long pathIndex = entry.getKey() | 1 << _branches;
				elseBranches.put(pathIndex, new LineageMap(entry.getValue()));
			}
			_distinctPaths.putAll(elseBranches);
		}
		_branches++;
	}
}

