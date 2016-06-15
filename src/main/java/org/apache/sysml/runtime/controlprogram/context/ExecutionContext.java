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

package org.apache.sysml.runtime.controlprogram.context;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.debug.DMLFrame;
import org.apache.sysml.debug.DMLProgramCounter;
import org.apache.sysml.debug.DebugState;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.caching.CacheException;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.caching.FrameObject;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.cp.BooleanObject;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.DoubleObject;
import org.apache.sysml.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysml.runtime.instructions.cp.IntObject;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.instructions.cp.StringObject;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixDimensionsMetaData;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.MetaData;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.runtime.util.UtilFunctions;


public class ExecutionContext 
{
	
	//program reference (e.g., function repository)
	protected Program _prog = null;
	
	//symbol table
	protected LocalVariableMap _variables;
	
	//debugging (optional)
	protected DebugState _dbState = null;
	
	protected GPUContext _gpuCtx = null;
	
	protected ExecutionContext()
	{
		//protected constructor to force use of ExecutionContextFactory
		this( true, null );
	}
	
	protected ExecutionContext(Program prog)
	{
		//protected constructor to force use of ExecutionContextFactory
		this( true, prog );
	}
	
	protected ExecutionContext(LocalVariableMap vars)
	{
		//protected constructor to force use of ExecutionContextFactory
		this( false, null);
		_variables = vars;
	}

	protected ExecutionContext( boolean allocateVariableMap, Program prog )
	{
		//protected constructor to force use of ExecutionContextFactory
		if( allocateVariableMap )
			_variables = new LocalVariableMap();
		else
			_variables = null;
		_prog = prog;
		if (DMLScript.ENABLE_DEBUG_MODE){
			_dbState = DebugState.getInstance();
		}
	}
	
	public Program getProgram(){
		return _prog;
	}
	
	public LocalVariableMap getVariables() {
		return _variables;
	}
	
	public void setVariables(LocalVariableMap vars) {
		_variables = vars;
	}
	
	public void destroyGPUContext() throws DMLRuntimeException {
		if(_gpuCtx != null)
			_gpuCtx.destroy();
	}
	
	
	/* -------------------------------------------------------
	 * Methods to handle variables and associated data objects
	 * -------------------------------------------------------
	 */
	
	public Data getVariable(String name) 
	{
		return _variables.get(name);
	}
	
	public void setVariable(String name, Data val) 
		throws DMLRuntimeException
	{
		_variables.put(name, val);
	}

	public Data removeVariable(String name) 
	{
		return _variables.remove(name);
	}

	public void setMetaData(String fname, MetaData md) 
		throws DMLRuntimeException 
	{
		_variables.get(fname).setMetaData(md);
	}
	
	public MetaData getMetaData(String varname) 
		throws DMLRuntimeException 
	{
		return _variables.get(varname).getMetaData();
	}
	
	public void removeMetaData(String varname) 
		throws DMLRuntimeException 
	{
		_variables.get(varname).removeMetaData();
	}
	
	public MatrixObject getMatrixObject(String varname) 
		throws DMLRuntimeException
	{
		Data dat = getVariable(varname);
		
		//error handling if non existing or no matrix
		if( dat == null )
			throw new DMLRuntimeException("Variable '"+varname+"' does not exist in the symbol table.");
		if( !(dat instanceof MatrixObject) )
			throw new DMLRuntimeException("Variable '"+varname+"' is not a matrix.");
		
		return (MatrixObject) dat;
	}
	
	public FrameObject getFrameObject(String varname) 
		throws DMLRuntimeException
	{
		Data dat = getVariable(varname);
		
		//error handling if non existing or no matrix
		if( dat == null )
			throw new DMLRuntimeException("Variable '"+varname+"' does not exist in the symbol table.");
		if( !(dat instanceof FrameObject) )
			throw new DMLRuntimeException("Variable '"+varname+"' is not a frame.");
		
		return (FrameObject) dat;
	}
	
	/**
	 * 
	 * @param varname
	 * @return
	 * @throws DMLRuntimeException
	 */
	public CacheableData<?> getCacheableData(String varname) 
		throws DMLRuntimeException
	{
		Data dat = getVariable(varname);
		
		//error handling if non existing or no matrix
		if( dat == null )
			throw new DMLRuntimeException("Variable '"+varname+"' does not exist in the symbol table.");
		if( !(dat instanceof CacheableData<?>) )
			throw new DMLRuntimeException("Variable '"+varname+"' is not a matrix or frame.");
		
		return (CacheableData<?>) dat;
	}
	
	/**
	 * 
	 * @param varname
	 * @throws DMLRuntimeException
	 */
	public void releaseCacheableData(String varname) 
		throws DMLRuntimeException
	{
		CacheableData<?> dat = getCacheableData(varname);
		dat.release();
	}
	
	public MatrixCharacteristics getMatrixCharacteristics( String varname ) 
		throws DMLRuntimeException
	{
		MatrixDimensionsMetaData dims = (MatrixDimensionsMetaData) getMetaData(varname);
		return dims.getMatrixCharacteristics();
	}
	
	/**
	 * Pins a matrix variable into memory and returns the internal matrix block.
	 * 
	 * @param varName
	 * @return
	 * @throws DMLRuntimeException
	 */
	public MatrixBlock getMatrixInput(String varName) 
		throws DMLRuntimeException 
	{	
		MatrixObject mo = getMatrixObject(varName);
		MatrixBlock mb = mo.acquireRead();
		return mb;
	}
	
	public void setMetaData(String varName, long nrows, long ncols) throws DMLRuntimeException  {
		MatrixObject mo = getMatrixObject(varName);
		if(mo.getNumRows() != nrows || mo.getNumColumns() != ncols) {
			MatrixCharacteristics mc = new MatrixCharacteristics((long)nrows, (long)ncols, 
					(int) mo.getNumRowsPerBlock(), (int)mo.getNumColumnsPerBlock());
			OutputInfo oiOld = null;
			InputInfo iiOld = null;
			MetaData oldMetaData = mo.getMetaData();
			if(oldMetaData != null && oldMetaData instanceof MatrixFormatMetaData) {
				oiOld = ((MatrixFormatMetaData)oldMetaData).getOutputInfo();
				iiOld = ((MatrixFormatMetaData)oldMetaData).getInputInfo();
			}
			else {
				throw new DMLRuntimeException("Metadata not available");
			}
			mo.setMetaData(new MatrixFormatMetaData(mc, oiOld, iiOld));
		}
	}
	
	public MatrixObject getMatrixOutputForGPUInstruction(String varName, boolean isSparse) 
			throws DMLRuntimeException {	
		if(isSparse) {
			throw new DMLRuntimeException("Sparse matrix block is not supported for GPU instruction");
		}
		MatrixObject mo = getMatrixObject(varName);
		if(mo.getMatrixBlock() == null) {
			MatrixBlock mb = new MatrixBlock((int)mo.getNumRows(), (int)mo.getNumColumns(), false);
			mo.acquireModify(mb);
			mo.release();
		}
		mo.getGPUObject().acquireDenseDeviceModify((int)(mo.getNumRows()*mo.getNumColumns()));
		mo.getMatrixCharacteristics().setNonZeros(-1);
		mo.getMatrixBlock().setNonZeros(-1);
		return mo;
	}
	
	public MatrixObject getMatrixInputForGPUInstruction(String varName) 
			throws DMLRuntimeException 
	{	
		MatrixObject mo = getMatrixObject(varName);
		if(mo == null) {
			throw new DMLRuntimeException("No matrix object available for variable:" + varName);
		}
		if(mo.getGPUObject() == null || !mo.getGPUObject().isAllocated) {
			mo.acquireRead();
			mo.release();
		}
		mo.getGPUObject().acquireDeviceRead();
		return mo;
	}
	
	/**
	 * Unpins a currently pinned matrix variable. 
	 * 
	 * @param varName
	 * @throws DMLRuntimeException
	 */
	public void releaseMatrixInput(String varName) 
		throws DMLRuntimeException 
	{
		MatrixObject mo = getMatrixObject(varName);
		mo.release();
	}
	
	public void releaseMatrixInputForGPUInstruction(String varName) 
			throws DMLRuntimeException 
		{
			MatrixObject mo = getMatrixObject(varName);
			mo.getGPUObject().release(false);
		}
	
	/**
	 * Pins a frame variable into memory and returns the internal frame block.
	 * 
	 * @param varName
	 * @return
	 * @throws DMLRuntimeException
	 */
	public FrameBlock getFrameInput(String varName) 
		throws DMLRuntimeException 
	{	
		FrameObject fo = getFrameObject(varName);
		return fo.acquireRead();
	}
	
	/**
	 * Unpins a currently pinned frame variable. 
	 * 
	 * @param varName
	 * @throws DMLRuntimeException
	 */
	public void releaseFrameInput(String varName) 
		throws DMLRuntimeException 
	{
		FrameObject fo = getFrameObject(varName);
		fo.release();
	}
	
	public ScalarObject getScalarInput(String name, ValueType vt, boolean isLiteral)
		throws DMLRuntimeException 
	{
		if ( isLiteral ) {
			switch (vt) {
			case INT:
				long intVal = UtilFunctions.parseToLong(name);				
				IntObject intObj = new IntObject(intVal);
				return intObj;
			case DOUBLE:
				double doubleVal = Double.parseDouble(name);
				DoubleObject doubleObj = new DoubleObject(doubleVal);
				return doubleObj;
			case BOOLEAN:
				Boolean boolVal = Boolean.parseBoolean(name);
				BooleanObject boolObj = new BooleanObject(boolVal);
				return boolObj;
			case STRING:
				StringObject stringObj = new StringObject(name);
				return stringObj;
			default:
				throw new DMLRuntimeException("Unknown value type: " + vt + " for variable: " + name);
			}
		}
		else {
			Data obj = getVariable(name);
			if (obj == null) {
				throw new DMLRuntimeException("Unknown variable: " + name);
			}
			return (ScalarObject) obj;
		}
	}

	public void setScalarOutput(String varName, ScalarObject so) 
		throws DMLRuntimeException 
	{
		setVariable(varName, so);
	}
	
	public void releaseMatrixOutputForGPUInstruction(String varName) throws DMLRuntimeException {
		MatrixObject mo = getMatrixObject(varName);
		if(mo.getGPUObject() == null || !mo.getGPUObject().isAllocated) {
			throw new DMLRuntimeException("No output is allocated on GPU");
		}
		mo.getGPUObject().release(true);
//		mo.acquireModify();
//		mo.release();
	}
	
	/**
	 * 
	 * @param varName
	 * @param outputData
	 * @throws DMLRuntimeException
	 */
	public void setMatrixOutput(String varName, MatrixBlock outputData) 
			throws DMLRuntimeException 
	{
		MatrixObject mo = getMatrixObject(varName);
		if(mo.getGPUObject() != null && mo.getGPUObject().isAllocated) {
			throw new DMLRuntimeException("GPU instructions should not set matrix output. "
					+ "Instead should use releaseMatrixOutput. If called by non-GPU instruction, "
					+ "then inconsistent bufferpool logic. Possible skipped deleting GPU object when acquire modify.");
		}
		
		mo.acquireModify(outputData);
	    mo.release();
	    setVariable(varName, mo);
	}
	
	/**
	 * 
	 * @param varName
	 * @param outputData
	 * @param inplace
	 * @throws DMLRuntimeException
	 */
	public void setMatrixOutput(String varName, MatrixBlock outputData, UpdateType flag) 
		throws DMLRuntimeException 
	{
		if( flag.isInPlace() ) {
			//modify metadata to carry update status
			MatrixObject mo = getMatrixObject(varName);
			mo.setUpdateType( flag );
		}
		
		//default case
		setMatrixOutput(varName, outputData);
	}
	
	/**
	 * 
	 * @param varName
	 * @param outputData
	 * @throws DMLRuntimeException
	 */
	public void setFrameOutput(String varName, FrameBlock outputData) 
		throws DMLRuntimeException 
	{
		FrameObject fo = getFrameObject(varName);
		if( outputData.getNumColumns()>0 && outputData.getSchema()!=null )
			fo.setValueType(outputData.getSchema().get(0));
		fo.acquireModify(outputData);
		fo.release();
		    
	    setVariable(varName, fo);
	}
	
	/**
	 * Pin a given list of variables i.e., set the "clean up" state in 
	 * corresponding matrix objects, so that the cached data inside these
	 * objects is not cleared and the corresponding HDFS files are not 
	 * deleted (through rmvar instructions). 
	 * 
	 * This is necessary for: function input variables, parfor result variables, 
	 * parfor shared inputs that are passed to functions.
	 * 
	 * The function returns the OLD "clean up" state of matrix objects.
	 */
	public HashMap<String,Boolean> pinVariables(ArrayList<String> varList) 
	{
		//2-pass approach since multiple vars might refer to same matrix object
		HashMap<String, Boolean> varsState = new HashMap<String,Boolean>();
		
		//step 1) get current information
		for( String var : varList )
		{
			Data dat = _variables.get(var);
			if( dat instanceof MatrixObject )
			{
				MatrixObject mo = (MatrixObject)dat;
				varsState.put( var, mo.isCleanupEnabled() );
				//System.out.println("pre-pin "+var+" ("+mo.isCleanupEnabled()+")");
			}
		}
		
		//step 2) pin variables
		for( String var : varList )
		{
			Data dat = _variables.get(var);
			if( dat instanceof MatrixObject )
			{
				MatrixObject mo = (MatrixObject)dat;
				mo.enableCleanup(false); 
				//System.out.println("pin "+var);
			}
		}
		
		return varsState;
	}
	
	/**
	 * Unpin the a given list of variables by setting their "cleanup" status
	 * to the values specified by <code>varsStats</code>.
	 * 
	 * Typical usage:
	 *    <code> 
	 *    oldStatus = pinVariables(varList);
	 *    ...
	 *    unpinVariables(varList, oldStatus);
	 *    </code>
	 * 
	 * i.e., a call to unpinVariables() is preceded by pinVariables(). 
	 */
	public void unpinVariables(ArrayList<String> varList, HashMap<String,Boolean> varsState)
	{
		for( String var : varList)
		{
			//System.out.println("unpin "+var+" ("+varsState.get(var)+")");
			Data dat = _variables.get(var);
			if( dat instanceof MatrixObject )
				((MatrixObject)dat).enableCleanup(varsState.get(var));
		}
	}
	
	/**
	 * NOTE: No order guaranteed, so keep same list for pin and unpin. 
	 * 
	 * @return
	 */
	public ArrayList<String> getVarList()
	{
		ArrayList<String> varlist = new ArrayList<String>();
		varlist.addAll(_variables.keySet());	
		return varlist;
	}

	
	/**
	 * 
	 * @param mo
	 * @throws CacheException
	 * @throws IOException
	 */
	public void cleanupMatrixObject(MatrixObject mo) 
		throws DMLRuntimeException 
	{
		try
		{
			if ( mo.isCleanupEnabled() ) 
			{
				//compute ref count only if matrix cleanup actually necessary
				if ( !getVariables().hasReferences(mo) ) {
					//clean cached data	
					mo.clearData(); 
					if( mo.isHDFSFileExists() )
					{
						//clean hdfs data
						String fpath = mo.getFileName();
						if (fpath != null) {
							MapReduceTool.deleteFileIfExistOnHDFS(fpath);
							MapReduceTool.deleteFileIfExistOnHDFS(fpath + ".mtd");
						}
					}
				}
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
	}
	
	
	///////////////////////////////
	// Debug State Functionality
	///////////////////////////////
	
	public void initDebugProgramCounters() 
	{
		if (DMLScript.ENABLE_DEBUG_MODE){
			_dbState.pc = new DMLProgramCounter(DMLProgram.DEFAULT_NAMESPACE, "main", 0, 0); //initialize program counter (pc)
			_dbState.prevPC = new DMLProgramCounter(DMLProgram.DEFAULT_NAMESPACE, "main", 0, 0); //initialize previous pc
		}
	}
	
	/**
	 * 
	 * @param index
	 * @throws DMLRuntimeException 
	 */
	public void updateDebugState( int index ) throws DMLRuntimeException 
	{
		if(DMLScript.ENABLE_DEBUG_MODE) {
			_dbState.getPC().setProgramBlockNumber(index);
		}
	}
	
	/**
	 * 
	 * @param currInst
	 * @throws DMLRuntimeException 
	 */
	public void updateDebugState( Instruction currInst ) throws DMLRuntimeException
	{
		if (DMLScript.ENABLE_DEBUG_MODE) {
			// New change so that shell doesnot seem like it is hanging while running MR job
			// Since UI cannot accept instructions when user is submitting the program
			_dbState.nextCommand = false;
			// Change to stop before first instruction of a given line
			//update current instruction ID and line number 
			_dbState.getPC().setInstID(currInst.getInstID()); 				
			_dbState.getPC().setLineNumber(currInst.getLineNum());
			// Change to stop before first instruction of a given line
			suspendIfAskedInDebugMode(currInst);	
		}
	}
	
	/**
	 * 
	 */
	public void clearDebugProgramCounters()
	{
		if(DMLScript.ENABLE_DEBUG_MODE) {
			_dbState.pc = null;
		}
	}
	
	public void handleDebugException( Exception ex )
	{
		_dbState.getDMLStackTrace(ex);
		_dbState.suspend = true;
	}

	public void handleDebugFunctionEntry( FunctionCallCPInstruction funCallInst ) throws DMLRuntimeException
	{
		//push caller frame into call stack
		_dbState.pushFrame(getVariables(), _dbState.getPC());
		//initialize pc for callee's frame
		_dbState.pc = new DMLProgramCounter(funCallInst.getNamespace(), funCallInst.getFunctionName(), 0, 0);
	}
	
	public void handleDebugFunctionExit( FunctionCallCPInstruction funCallInst )
	{
		//pop caller frame from call stack
		DMLFrame fr = _dbState.popFrame();
		//update pc to caller's frame
		_dbState.pc = fr.getPC();
	}
	
	public DebugState getDebugState() {
		return _dbState;
	}
	
	/**
	 * This function should be called only if user has specified -debug option.
	 * In this function, if the user has issued one of the step instructions or
	 * has enabled suspend flag in previous instruction (through breakpoint),
	 * then it will wait until user issues a new debugger command.
	 * @param currInst
	 * @param ec
	 * @throws DMLRuntimeException 
	 */
	@SuppressWarnings("deprecation")
	private void suspendIfAskedInDebugMode(Instruction currInst ) throws DMLRuntimeException {
		if (!DMLScript.ENABLE_DEBUG_MODE) {
			System.err.println("ERROR: The function suspendIfAskedInDebugMode should not be called in non-debug mode.");
		}
		//check for stepping options
		if (!_dbState.suspend && _dbState.dbCommand != null) { 
			if (_dbState.dbCommand.equalsIgnoreCase("step_instruction")) {
				System.out.format("Step instruction reached at %s.\n", _dbState.getPC().toString());
				_dbState.suspend = true;
			}
			else if (_dbState.dbCommand.equalsIgnoreCase("step_line") && _dbState.prevPC.getLineNumber() != currInst.getLineNum()
					&& _dbState.prevPC.getLineNumber() != 0) {
				// Don't step into first instruction of first line
				// System.out.format("Step reached at %s.\n", this._prog.getPC().toString());
				System.out.format("Step reached at %s.\n", _dbState.getPC().toStringWithoutInstructionID());
				_dbState.suspend = true;
			}
			else if (_dbState.dbCommand.equalsIgnoreCase("step return") && currInst instanceof FunctionCallCPInstruction) {
				FunctionCallCPInstruction funCallInst = (FunctionCallCPInstruction) currInst;
				if (_dbState.dbCommandArg == null || funCallInst.getFunctionName().equalsIgnoreCase(_dbState.dbCommandArg)) {
					System.out.format("Step return reached at %s.\n", _dbState.getPC().toStringWithoutInstructionID());
					_dbState.suspend = true;
				}
			}
		}
		//check if runtime suspend signal is set
		if (_dbState.suspend) {
			//flush old commands and arguments
			_dbState.dbCommand = null;
			_dbState.dbCommandArg = null;
			//print current DML script source line
			if (currInst.getLineNum() != 0)
				_dbState.printDMLSourceLine(currInst.getLineNum());
			//save current symbol table
			_dbState.setVariables(this.getVariables());
			//send next command signal to debugger control module
			_dbState.nextCommand = true;
			//suspend runtime execution thread
			Thread.currentThread().suspend();
			//reset next command signal
			_dbState.nextCommand = false;
		}
		//reset runtime suspend signal
		_dbState.suspend = false;
		//update previous pc
		_dbState.prevPC.setFunctionName(_dbState.getPC().getFunctionName());
		_dbState.prevPC.setProgramBlockNumber(_dbState.getPC().getProgramBlockNumber());
		_dbState.prevPC.setLineNumber(currInst.getLineNum());
	}
}
