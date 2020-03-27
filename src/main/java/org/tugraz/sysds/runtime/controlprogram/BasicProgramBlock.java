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

package org.tugraz.sysds.runtime.controlprogram;

import java.util.ArrayList;

import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.hops.recompile.Recompiler;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.instructions.Instruction;
import org.tugraz.sysds.runtime.lineage.LineageCache;
import org.tugraz.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.tugraz.sysds.runtime.lineage.LineageCacheStatistics;
import org.tugraz.sysds.runtime.lineage.LineageItem;
import org.tugraz.sysds.runtime.lineage.LineageItemUtils;
import org.tugraz.sysds.utils.Statistics;

public class BasicProgramBlock extends ProgramBlock 
{
	protected ArrayList<Instruction> _inst;

	public BasicProgramBlock(Program prog) {
		super(prog);
		_inst = new ArrayList<>();
	}

	public  ArrayList<Instruction> getInstructions() {
		return _inst;
	}

	public Instruction getInstruction(int i) {
		return _inst.get(i);
	}

	public  void setInstructions( ArrayList<Instruction> inst ) {
		_inst = inst;
	}

	public void addInstruction(Instruction inst) {
		_inst.add(inst);
	}

	public void addInstructions(ArrayList<Instruction> inst) {
		_inst.addAll(inst);
	}

	public int getNumInstructions() {
		return _inst.size();
	}
	
	@Override
	public ArrayList<ProgramBlock> getChildBlocks() {
		return null;
	}
	
	@Override
	public boolean isNested() {
		return false;
	}

	@Override
	public void execute(ExecutionContext ec)
	{
		ArrayList<Instruction> tmp = _inst;

		//dynamically recompile instructions if enabled and required
		try
		{
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			if( ConfigurationManager.isDynamicRecompilation()
				&& _sb != null
				&& _sb.requiresRecompilation() )
			{
				tmp = Recompiler.recompileHopsDag(
					_sb, _sb.getHops(), ec, null, false, true, _tid);
			}
			if( DMLScript.STATISTICS ){
				long t1 = System.nanoTime();
				Statistics.incrementHOPRecompileTime(t1-t0);
				if( tmp!=_inst )
					Statistics.incrementHOPRecompileSB();
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Unable to recompile program block.", ex);
		}
		
		//statement-block-level, lineage-based reuse
		LineageItem[] liInputs = null;
		if (_sb != null && !ReuseCacheType.isNone()) {
			String name = "SB" + _sb.getSBID();
			liInputs = LineageItemUtils.getLineageItemInputstoSB(_sb.getInputstoSB(), ec);
			if( LineageCache.reuse(_sb.getOutputsofSB(), _sb.getOutputsofSB().size(), liInputs, name, ec) ) {
				if( DMLScript.STATISTICS )
					LineageCacheStatistics.incrementSBHits();
				return;
			}
		}

		//actual instruction execution
		executeInstructions(tmp, ec);
		
		//statement-block-level, lineage-based caching
		if (_sb != null && liInputs != null) {
			String name = "SB" + _sb.getSBID();
			LineageCache.putValue(_sb.getOutputsofSB(), _sb.getOutputsofSB().size(), liInputs, name, ec);
		}
	}
}
