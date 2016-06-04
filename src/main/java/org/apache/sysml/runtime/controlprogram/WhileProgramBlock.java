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

package org.apache.sysml.runtime.controlprogram;

import java.util.ArrayList;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.parser.WhileStatementBlock;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLScriptException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.Instruction.INSTRUCTION_TYPE;
import org.apache.sysml.runtime.instructions.cp.BooleanObject;
import org.apache.sysml.runtime.instructions.cp.CPInstruction;
import org.apache.sysml.runtime.instructions.cp.ComputationCPInstruction;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.instructions.cp.StringObject;
import org.apache.sysml.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysml.runtime.instructions.cp.CPInstruction.CPINSTRUCTION_TYPE;
import org.apache.sysml.yarn.DMLAppMasterUtils;


public class WhileProgramBlock extends ProgramBlock 
{
	
	private ArrayList<Instruction> _predicate;
	private String _predicateResultVar;
	private ArrayList <Instruction> _exitInstructions ;
	private ArrayList<ProgramBlock> _childBlocks;

	public WhileProgramBlock(Program prog, ArrayList<Instruction> predicate) throws DMLRuntimeException{
		super(prog);
		_predicate = predicate;
		_predicateResultVar = findPredicateResultVar ();
		_exitInstructions = new ArrayList<Instruction>();
		_childBlocks = new ArrayList<ProgramBlock>(); 
	}
	
	public void addProgramBlock(ProgramBlock childBlock) {
		_childBlocks.add(childBlock);
	}
	
	public void setExitInstructions2(ArrayList<Instruction> exitInstructions) { 
		_exitInstructions = exitInstructions; 
	}

	public void setExitInstructions1(ArrayList<Instruction> predicate) { 
		_predicate = predicate; 
	}
	
	public void addExitInstruction(Instruction inst) { 
		_exitInstructions.add(inst); 
	}
	
	public ArrayList<Instruction> getPredicate() { 
		return _predicate; 
	}
	
	public void setPredicate( ArrayList<Instruction> predicate ) { 
		_predicate = predicate;
		
		//update result var if non-empty predicate (otherwise,
		//do not overwrite varname predicate in predicateResultVar)
		if( _predicate != null && !_predicate.isEmpty()  )
			_predicateResultVar = findPredicateResultVar();
	}
	
	public String getPredicateResultVar() { 
		return _predicateResultVar; 
	}
	
	public void setPredicateResultVar(String resultVar) { 
		_predicateResultVar = resultVar; 
	}
	
	public ArrayList<Instruction> getExitInstructions() { 
		return _exitInstructions; 
	}
	
	private BooleanObject executePredicate(ExecutionContext ec) 
		throws DMLRuntimeException 
	{
		BooleanObject result = null;
		try
		{
			if( _predicate!=null && !_predicate.isEmpty() )
			{
				if( _sb!=null )
				{
					if( DMLScript.isActiveAM() ) //set program block specific remote memory
						DMLAppMasterUtils.setupProgramBlockRemoteMaxMemory(this);
					
					WhileStatementBlock wsb = (WhileStatementBlock)_sb;
					Hop predicateOp = wsb.getPredicateHops();
					boolean recompile = wsb.requiresPredicateRecompilation();
					result = (BooleanObject) executePredicate(_predicate, predicateOp, recompile, ValueType.BOOLEAN, ec);
				}
				else
					result = (BooleanObject) executePredicate(_predicate, null, false, ValueType.BOOLEAN, ec);
			}
			else 
			{
				//get result var
				ScalarObject scalarResult = null;
				Data resultData = ec.getVariable(_predicateResultVar);
				if ( resultData == null ) {
					//note: resultvar is a literal (can it be of any value type other than String, hence no literal/varname conflict) 
					scalarResult = ec.getScalarInput(_predicateResultVar, ValueType.BOOLEAN, true);
				}
				else {
					scalarResult = ec.getScalarInput(_predicateResultVar, ValueType.BOOLEAN, false);
				}
				
				//check for invalid type String 
				if (scalarResult instanceof StringObject)
					throw new DMLRuntimeException(this.printBlockErrorLocation() + "\nWhile predicate variable "+ _predicateResultVar + " evaluated to string " + scalarResult + " which is not allowed for predicates in DML");
				
				//process result
				if( scalarResult instanceof BooleanObject )
					result = (BooleanObject)scalarResult;
				else
					result = new BooleanObject( scalarResult.getBooleanValue() ); //auto casting
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(this.printBlockErrorLocation() + "Failed to evaluate the while predicate.", ex);
		}
		
		//(guaranteed to be non-null, see executePredicate/getScalarInput)
		return result;
	}
	
	public void execute(ExecutionContext ec) throws DMLRuntimeException 
	{
		//execute while loop
		try 
		{
			// prepare update in-place variables
			UpdateType[] flags = prepareUpdateInPlaceVariables(ec);
			
			//run loop body until predicate becomes false
			while( executePredicate(ec).getBooleanValue() )
			{		
				//execute all child blocks
				for (int i=0 ; i < _childBlocks.size() ; i++) {
					ec.updateDebugState(i);
					_childBlocks.get(i).execute(ec);
				}
			}
			
			// reset update-in-place variables
			resetUpdateInPlaceVariableFlags(ec, flags);
		}
		catch (DMLScriptException e) {
			//propagate stop call
			throw e;
		}
		catch (Exception e) {
			throw new DMLRuntimeException(printBlockErrorLocation() + "Error evaluating while program block", e);
		}
		
		//execute exit instructions
		try {
			executeInstructions(_exitInstructions, ec);
		}
		catch(Exception e) {
			throw new DMLRuntimeException(printBlockErrorLocation() + "Error executing while exit instructions.", e);
		}
	}

	public ArrayList<ProgramBlock> getChildBlocks() {
		return _childBlocks;
	}
	
	public void setChildBlocks(ArrayList<ProgramBlock> childs) {
		_childBlocks = childs;
	}
	
	private String findPredicateResultVar ( ) {
		String result = null;
		for ( Instruction si : _predicate ) {
			if ( si.getType() == INSTRUCTION_TYPE.CONTROL_PROGRAM && ((CPInstruction)si).getCPInstructionType() != CPINSTRUCTION_TYPE.Variable ) {
				result = ((ComputationCPInstruction) si).getOutputVariableName();  
			}
			else if(si instanceof VariableCPInstruction && ((VariableCPInstruction)si).isVariableCastInstruction()){
				result = ((VariableCPInstruction)si).getOutputVariableName();
			}
		}
		
		return result;
	}
	
	public String printBlockErrorLocation(){
		return "ERROR: Runtime error in while program block generated from while statement block between lines " + _beginLine + " and " + _endLine + " -- ";
	}
}