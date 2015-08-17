/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.hops.recompile.Recompiler;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLScriptException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.cp.Data;


public class FunctionProgramBlock extends ProgramBlock 
{
	
	protected ArrayList<ProgramBlock> _childBlocks;
	protected ArrayList<DataIdentifier> _inputParams;
	protected ArrayList<DataIdentifier> _outputParams;
	
	private boolean _recompileOnce = false;
	
	public FunctionProgramBlock( Program prog, ArrayList<DataIdentifier> inputParams, ArrayList<DataIdentifier> outputParams) throws DMLRuntimeException
	{
		super(prog);
		_childBlocks = new ArrayList<ProgramBlock>();
		_inputParams = new ArrayList<DataIdentifier>();
		for (DataIdentifier id : inputParams){
			_inputParams.add(new DataIdentifier(id));
			
		}
		_outputParams = new ArrayList<DataIdentifier>();
		for (DataIdentifier id : outputParams){
			_outputParams.add(new DataIdentifier(id));
		}
	}
	
	public ArrayList<DataIdentifier> getInputParams(){
		return _inputParams;
	}
	
	public ArrayList<DataIdentifier> getOutputParams(){
		return _outputParams;
	}
	
	public void addProgramBlock(ProgramBlock childBlock) {
		_childBlocks.add(childBlock);
	}
	
	public void setChildBlocks( ArrayList<ProgramBlock> pbs)
	{
		_childBlocks = pbs;
	}
	
	public ArrayList<ProgramBlock> getChildBlocks() {
		return _childBlocks;
	}
	
	@Override
	public void execute(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{	
		//dynamically recompile entire function body (according to function inputs)
		try {
			if( OptimizerUtils.ALLOW_DYN_RECOMPILATION 
				&& isRecompileOnce() 
				&& ParForProgramBlock.RESET_RECOMPILATION_FLAGs )
			{
				//note: it is important to reset the recompilation flags here
				// (1) it is safe to reset recompilation flags because a 'recompile_once'
				//     function will be recompiled for every execution.
				// (2) without reset, there would be no benefit in recompiling the entire function
				LocalVariableMap tmp = (LocalVariableMap) ec.getVariables().clone();
				Recompiler.recompileProgramBlockHierarchy(_childBlocks, tmp, _tid, true);
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException("Error recompiling function body.", ex);
		}
		
		// for each program block
		try {						
			for (int i=0 ; i < this._childBlocks.size() ; i++) {
				ec.updateDebugState(i);
				_childBlocks.get(i).execute(ec);
			}
		}
		catch (DMLScriptException e) {
			throw e;
		}
		catch (Exception e){
			throw new DMLRuntimeException(this.printBlockErrorLocation() + "Error evaluating function program block", e);
		}
		
		// check return values
		checkOutputParameters(ec.getVariables());
	}
	
	/**
	 * 
	 * @param vars
	 */
	protected void checkOutputParameters( LocalVariableMap vars )
	{
		for( DataIdentifier diOut : _outputParams )
		{
			String varName = diOut.getName();
			Data dat = vars.get( varName );
			if( dat == null )
				LOG.error("Function output "+ varName +" is missing.");
			else if( dat.getDataType() != diOut.getDataType() )
				LOG.warn("Function output "+ varName +" has wrong data type: "+dat.getDataType()+".");
			else if( dat.getValueType() != diOut.getValueType() )
				LOG.warn("Function output "+ varName +" has wrong value type: "+dat.getValueType()+".");
			   
		}
	}
	
	public void setRecompileOnce( boolean flag ) {
		_recompileOnce = flag;
	}
	
	public boolean isRecompileOnce() {
		return _recompileOnce;
	}
	
	public void printMe() {
		
		for (ProgramBlock pb : this._childBlocks){
			pb.printMe();
		}
	}
	
	public String printBlockErrorLocation(){
		return "ERROR: Runtime error in function program block generated from function statement block between lines " + _beginLine + " and " + _endLine + " -- ";
	}
	
}