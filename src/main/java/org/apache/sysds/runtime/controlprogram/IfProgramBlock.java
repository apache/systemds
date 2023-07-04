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

package org.apache.sysds.runtime.controlprogram;

import java.util.ArrayList;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.BooleanObject;


public class IfProgramBlock extends ProgramBlock 
{
	private ArrayList<Instruction> _predicate;
	private ArrayList<ProgramBlock> _childBlocksIfBody;
	private ArrayList<ProgramBlock> _childBlocksElseBody;
	private int _lineagePathPos = -1;
	
	public IfProgramBlock(Program prog, ArrayList<Instruction> predicate) {
		super(prog);
		_childBlocksIfBody = new ArrayList<>();
		_childBlocksElseBody = new ArrayList<>();
		_predicate = predicate;
	}
	
	public ArrayList<ProgramBlock> getChildBlocksIfBody() { 
		return getChildBlocks();
	}

	public void setChildBlocksIfBody(ArrayList<ProgramBlock> blocks) { 
		_childBlocksIfBody = blocks; 
	}
	
	public void addProgramBlockIfBody(ProgramBlock pb) { 
		_childBlocksIfBody.add(pb); 
	}	
	
	public ArrayList<ProgramBlock> getChildBlocksElseBody() { 
		return _childBlocksElseBody; 
	}

	public void setChildBlocksElseBody(ArrayList<ProgramBlock> blocks) { 
		_childBlocksElseBody = blocks; 
	}
	
	public void addProgramBlockElseBody(ProgramBlock pb) {
		_childBlocksElseBody.add(pb); 
	}
	
	public ArrayList<Instruction> getPredicate(){
		return _predicate;
	}

	public void setPredicate(ArrayList<Instruction> predicate) {
		_predicate = predicate;
	}
	
	@Override
	public ArrayList<ProgramBlock> getChildBlocks() {
		return _childBlocksIfBody;
	}
	
	@Override
	public boolean isNested() {
		return true;
	}
	
	public void setLineageDedupPathPos(int pos) {
		_lineagePathPos = pos;
	}
	
	@Override
	public void execute(ExecutionContext ec) 
	{
		BooleanObject predResult = executePredicate(ec);
	
		if (DMLScript.LINEAGE_DEDUP)
			ec.getLineage().setDedupPathBranch(_lineagePathPos, predResult.getBooleanValue());
		
		//execute if statement
		if(predResult.getBooleanValue()) {
			try  {
				for (int i=0 ; i < _childBlocksIfBody.size() ; i++) {
					_childBlocksIfBody.get(i).execute(ec);
				}
			}
			catch(DMLScriptException e) {
				throw e;
			}
			catch(Exception e) {
				throw new DMLRuntimeException(this.printBlockErrorLocation() + "Error evaluating if statement body ", e);
			}
		}
		else {
			try {
				for (int i=0 ; i < _childBlocksElseBody.size() ; i++) {
					_childBlocksElseBody.get(i).execute(ec);
				}
			}
			catch(DMLScriptException e) {
				throw e;
			}
			catch(Exception e) {
				throw new DMLRuntimeException(this.printBlockErrorLocation() + "Error evaluating else statement body ", e);
			}
		}
		
		//execute exit instructions
		executeExitInstructions("if", ec);
	}

	private BooleanObject executePredicate(ExecutionContext ec) 
	{
		BooleanObject result;
		try {
			if( _sb != null ) {
				IfStatementBlock isb = (IfStatementBlock)_sb;
				result = (BooleanObject) executePredicate(_predicate, isb.getPredicateHops(), 
					isb.requiresPredicateRecompilation(), ValueType.BOOLEAN, ec);
			}
			else
				result = (BooleanObject) executePredicate(_predicate, null, false, ValueType.BOOLEAN, ec);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(this.printBlockErrorLocation() + "Failed to evaluate the IF predicate.", ex);
		}
		
		//(guaranteed to be non-null, see executePredicate/getScalarInput)
		return result;
	}
	
	@Override
	public String printBlockErrorLocation(){
		return "ERROR: Runtime error in if program block generated from if statement block between lines " + _beginLine + " and " + _endLine + " -- ";
	}
}
