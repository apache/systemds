package org.tugraz.sysds.runtime.lineage;

import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.*;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.instructions.Instruction;

import java.util.HashMap;
import java.util.Map;

public class LineageDedupBlock {
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
	
	public void traceIfProgramBlock(IfProgramBlock ipb, ExecutionContext ec) {
		addPathsForBranch();
		traceElseBodyInstructions(ipb, ec);
		traceIfBodyInstructions(ipb, ec);
		_activePath = null;
		
	}
	
	public void traceProgramBlock(ProgramBlock pb, ExecutionContext ec) {
		if (_distinctPaths.size() == 0)
			_distinctPaths.put(0L, new LineageMap());
		for (Map.Entry<Long, LineageMap> entry : _distinctPaths.entrySet())
			traceInstructions(pb, ec, entry);
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
		if (! (pb instanceof BasicProgramBlock) )
			throw new DMLRuntimeException("Only BasicProgramBLocks are allowed inside a LineageDedupBlock.");
		
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
