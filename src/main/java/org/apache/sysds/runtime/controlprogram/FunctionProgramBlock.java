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
import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FunctionBlock;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.hops.recompile.Recompiler.ResetType;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.util.ProgramConverter;
import org.apache.sysds.utils.Statistics;


public class FunctionProgramBlock extends ProgramBlock implements FunctionBlock
{
	public String _functionName;
	public String _namespace;
	protected ArrayList<ProgramBlock> _childBlocks;
	protected ArrayList<DataIdentifier> _inputParams;
	protected ArrayList<DataIdentifier> _outputParams;
	
	private boolean _recompileOnce = false;
	private boolean _nondeterministic = false;
	
	public FunctionProgramBlock( Program prog, List<DataIdentifier> inputParams, List<DataIdentifier> outputParams) {
		super(prog);
		_childBlocks = new ArrayList<>();
		_inputParams = new ArrayList<>();
		for (DataIdentifier id : inputParams)
			_inputParams.add(new DataIdentifier(id));
		_outputParams = new ArrayList<>();
		for (DataIdentifier id : outputParams)
			_outputParams.add(new DataIdentifier(id));
	}
	
	public DataIdentifier getInputParam(String name) {
		return _inputParams.stream()
			.filter(d -> d.getName().equals(name))
			.findFirst().orElse(null);
	}
	
	public List<String> getInputParamNames() {
		return _inputParams.stream().map(d -> d.getName()).collect(Collectors.toList());
	} 
	
	public List<String> getOutputParamNames() {
		return _outputParams.stream().map(d -> d.getName()).collect(Collectors.toList());
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
	
	public void setChildBlocks(ArrayList<ProgramBlock> pbs) {
		_childBlocks = pbs;
	}
	
	@Override
	public ArrayList<ProgramBlock> getChildBlocks() {
		return _childBlocks;
	}
	
	@Override
	public boolean isNested() {
		return true;
	}
	
	@Override
	public void execute(ExecutionContext ec) 
	{
		//dynamically recompile entire function body (according to function inputs)
		try {
			if( ConfigurationManager.isDynamicRecompilation() 
				&& isRecompileOnce() 
				&& ParForProgramBlock.RESET_RECOMPILATION_FLAGs )
			{
				long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
				
				//note: it is important to reset the recompilation flags here
				// (1) it is safe to reset recompilation flags because a 'recompile_once'
				//     function will be recompiled for every execution.
				// (2) without reset, there would be no benefit in recompiling the entire function
				LocalVariableMap tmp = (LocalVariableMap) ec.getVariables().clone();
				boolean codegen = ConfigurationManager.isCodegenEnabled();
				boolean singlenode = DMLScript.getGlobalExecMode() == ExecMode.SINGLE_NODE;
				ResetType reset = (codegen || singlenode) ? ResetType.RESET_KNOWN_DIMS : ResetType.RESET;
				Recompiler.recompileProgramBlockHierarchy(_childBlocks, tmp, _tid, false, reset);

				if( DMLScript.STATISTICS ){
					long t1 = System.nanoTime();
					Statistics.incrementFunRecompileTime(t1-t0);
					Statistics.incrementFunRecompiles();
				}
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException("Error recompiling function body.", ex);
		}
		
		// for each program block
		try {
			for (int i=0 ; i < this._childBlocks.size() ; i++) {
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

	protected void checkOutputParameters( LocalVariableMap vars )
	{
		for( DataIdentifier diOut : _outputParams ) {
			String varName = diOut.getName();
			Data dat = vars.get( varName );
			if( dat == null )
				LOG.error("Function output "+ varName +" is missing.");
			else if( dat.getDataType() != diOut.getDataType() )
				LOG.warn("Function output "+ varName +" has wrong data type: "+dat.getDataType()+".");
			else if( diOut.getValueType() != ValueType.UNKNOWN && dat.getValueType() != diOut.getValueType() )
				LOG.warn("Function output "+ varName +" has wrong value type: "+dat.getValueType()+".");
		}
	}
	
	public void setRecompileOnce( boolean flag ) {
		_recompileOnce = flag;
	}
	
	public boolean isRecompileOnce() {
		return _recompileOnce;
	}

	public void setNondeterministic(boolean flag) {
		_nondeterministic = flag;
	}
	
	public boolean isNondeterministic() {
		return _nondeterministic;
	}
	
	@Override
	public FunctionBlock cloneFunctionBlock() {
		return ProgramConverter
			.createDeepCopyFunctionProgramBlock(this, new HashSet<>(), new HashSet<>());
	}
	
	@Override
	public String printBlockErrorLocation(){
		return "ERROR: Runtime error in function program block generated from function statement block between lines " + _beginLine + " and " + _endLine + " -- ";
	}
}