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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.IfProgramBlock;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

public class LineageDedupBlock {
	private Map<Long, LineageMap> _distinctPaths = new HashMap<>();
	private BitSet _path = new BitSet();
	private int _numPaths = 0;
	
	private long _activePath = -1;
	private ArrayList<Long> _numDistinctPaths = new ArrayList<>();
	
	public LineageMap getActiveMap() {
		if (_activePath < 0 || !_distinctPaths.containsKey(_activePath))
			throw new DMLRuntimeException("Active path in LineageDedupBlock could not be found.");
		return _distinctPaths.get(_activePath);
	}
	
	public LineageMap getMap(Long path) {
		return _distinctPaths.containsKey(path) ? _distinctPaths.get(path) : null;
	}
	
	public Map<Long, LineageMap> getPathMaps() {
		return _distinctPaths;
	}
	
	public void setMap(Long takenPath, LineageMap tracedMap) {
		_distinctPaths.put(takenPath, new LineageMap(tracedMap));
	}
	
	public boolean pathExists(Long path) {
		return _distinctPaths.containsKey(path);
	}
	
	public void resetPath() {
		_path.clear();
	}
	
	public void setPathBranch(int pos, boolean value) {
		_path.set(pos, value);
	}
	
	public long getPath() {
		return _path.length() == 0 ? 0 :
			_path.toLongArray()[0];
	}
	
	public boolean isAllPathsTaken() {
		return _distinctPaths.size() == _numDistinctPaths.size();
	}
	
	public void traceProgramBlocks(ArrayList<ProgramBlock> pbs, ExecutionContext ec) {
		if (_distinctPaths.size() == 0) //main path
			_distinctPaths.put(0L, new LineageMap());
		//process top-level blocks with changing set of all paths
		//Note: list copy required as distinct paths is modified
		for( ProgramBlock pb : pbs )
			traceProgramBlock(pb, ec, new ArrayList<>(_distinctPaths.entrySet()));
	}
	
	public void traceProgramBlock(ProgramBlock pb, ExecutionContext ec, Collection<Entry<Long, LineageMap>> paths) {
		if (pb instanceof IfProgramBlock)
			traceIfProgramBlock((IfProgramBlock) pb, ec, paths);
		else if (pb instanceof BasicProgramBlock)
			traceBasicProgramBlock((BasicProgramBlock) pb, ec, paths);
		else
			throw new DMLRuntimeException("Only BasicProgramBlocks or "
				+ "IfProgramBlocks are allowed inside a LineageDedupBlock.");
	}
	
	public void traceIfProgramBlock(IfProgramBlock ipb, ExecutionContext ec, Collection<Entry<Long, LineageMap>> paths) {
		//step 0: materialize branch position
		ipb.setLineageDedupPathPos(_numPaths++);
		
		//step 1: create new paths
		//replicate all existing paths in current scope for the new branch
		//(existing path IDs reflect the else branch)
		Map<Long, LineageMap> rep = new HashMap<>();
		int pathKey = 1 << (_numPaths-1);
		for (Map.Entry<Long, LineageMap> entry : paths) {
			Long pathIndex = entry.getKey() | pathKey;
			rep.put(pathIndex, new LineageMap(entry.getValue()));
		}
		_distinctPaths.putAll(rep);
		
		//step 3: trace if and else branches separately
		for (ProgramBlock pb : ipb.getChildBlocksIfBody())
			traceProgramBlock(pb, ec, rep.entrySet());
		for (ProgramBlock pb : ipb.getChildBlocksElseBody())
			traceProgramBlock(pb, ec, paths);
	}
	
	public void traceBasicProgramBlock(BasicProgramBlock bpb, ExecutionContext ec, Collection<Entry<Long,LineageMap>> paths) {
		for (Entry<Long, LineageMap> entry : paths) {
			_activePath = entry.getKey();
			for (Instruction inst : bpb.getInstructions())
				entry.getValue().trace(inst, ec);
		}
	}
	// compute and save the number of distinct paths
	public void setNumPathsInPBs (ArrayList<ProgramBlock> pbs, ExecutionContext ec) {
		if (_numDistinctPaths.size() == 0) 
			_numDistinctPaths.add(0L);
		for (ProgramBlock pb : pbs)
			numPathsInPB(pb, ec, _numDistinctPaths);
	}
	
	private void numPathsInPB(ProgramBlock pb, ExecutionContext ec, ArrayList<Long> paths) {
		if (pb instanceof IfProgramBlock)
			numPathsInIfPB((IfProgramBlock)pb, ec, paths);
		else if (pb instanceof BasicProgramBlock)
			return;
		else
			throw new DMLRuntimeException("Only BasicProgramBlocks or "
				+ "IfProgramBlocks are allowed inside a LineageDedupBlock.");
	}
	
	private void numPathsInIfPB(IfProgramBlock ipb, ExecutionContext ec, ArrayList<Long> paths) {
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
