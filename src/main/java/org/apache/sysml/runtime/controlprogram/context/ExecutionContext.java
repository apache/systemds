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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.debug.DMLFrame;
import org.apache.sysml.debug.DMLProgramCounter;
import org.apache.sysml.debug.DebugState;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.caching.FrameObject;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.instructions.cp.ScalarObjectFactory;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.runtime.instructions.gpu.context.GPUObject;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixDimensionsMetaData;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.MetaData;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.util.MapReduceTool;


public class ExecutionContext {
	protected static final Log LOG = LogFactory.getLog(ExecutionContext.class.getName());

	//program reference (e.g., function repository)
	protected Program _prog = null;
	
	//symbol table
	protected LocalVariableMap _variables;
	
	//debugging (optional)
	protected DebugState _dbState = null;

	/**
	 * List of {@link GPUContext}s owned by this {@link ExecutionContext}
	 */
    protected List<GPUContext> _gpuContexts = new ArrayList<>();

	protected ExecutionContext()
	{
		//protected constructor to force use of ExecutionContextFactory
		this( true, null );
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

	/**
	 * Get the i-th GPUContext
	 * @param index index of the GPUContext
	 * @return a valid GPUContext or null if the indexed GPUContext does not exist.
	 */
    public GPUContext getGPUContext(int index) {
    	try {
			return _gpuContexts.get(index);
		} catch (IndexOutOfBoundsException e){
    		return null;
		}
    }

	/**
	 * Sets the list of GPUContexts
	 * @param gpuContexts a collection of GPUContexts
	 */
	public void setGPUContexts(List<GPUContext> gpuContexts){
		_gpuContexts = gpuContexts;
	}

	/**
	 * Gets the list of GPUContexts
	 * @return a list of GPUContexts
	 */
	public List<GPUContext> getGPUContexts() {
		return _gpuContexts;
	}

	/**
	 * Gets the number of GPUContexts
	 * @return number of GPUContexts
	 */
	public int getNumGPUContexts() {
    	return _gpuContexts.size();
	}

	/* -------------------------------------------------------
	 * Methods to handle variables and associated data objects
	 * -------------------------------------------------------
	 */
	
	public Data getVariable(String name) {
		return _variables.get(name);
	}
	
	public void setVariable(String name, Data val) {
		_variables.put(name, val);
	}
	
	public boolean containsVariable(String name) {
		return _variables.keySet().contains(name);
	}

	public Data removeVariable(String name) {
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
	 * @param varName variable name
	 * @return matrix block
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public MatrixBlock getMatrixInput(String varName) 
		throws DMLRuntimeException 
	{	
		MatrixObject mo = getMatrixObject(varName);
		return mo.acquireRead();
	}
	
	public void setMetaData(String varName, long nrows, long ncols) 
		throws DMLRuntimeException  
	{
		MatrixObject mo = getMatrixObject(varName);
		if(mo.getNumRows() == nrows && mo.getNumColumns() == ncols) 
			return;
		
		MetaData oldMetaData = mo.getMetaData();
		if( oldMetaData == null || !(oldMetaData instanceof MatrixFormatMetaData) )
			throw new DMLRuntimeException("Metadata not available");
			
		MatrixCharacteristics mc = new MatrixCharacteristics((long)nrows, (long)ncols, 
				(int) mo.getNumRowsPerBlock(), (int)mo.getNumColumnsPerBlock());
		mo.setMetaData(new MatrixFormatMetaData(mc, 
				((MatrixFormatMetaData)oldMetaData).getOutputInfo(),
				((MatrixFormatMetaData)oldMetaData).getInputInfo()));
	}

	/**
	 * Allocates a dense matrix on the GPU (for output)
	 * @param varName	name of the output matrix (known by this {@link ExecutionContext})
	 * @return a pair containing the wrapping {@link MatrixObject} and a boolean indicating whether a cuda memory allocation took place (as opposed to the space already being allocated)
	 * @throws DMLRuntimeException
	 */
	public Pair<MatrixObject, Boolean> getDenseMatrixOutputForGPUInstruction(String varName)
		throws DMLRuntimeException 
	{	
		MatrixObject mo = allocateGPUMatrixObject(varName);
		boolean allocated = mo.getGPUObject(getGPUContext(0)).acquireDeviceModifyDense();
		mo.getMatrixCharacteristics().setNonZeros(-1);
		return new Pair<MatrixObject, Boolean>(mo, allocated);
	}

    /**
     * Allocates a sparse matrix in CSR format on the GPU.
     * Assumes that mat.getNumRows() returns a valid number
     * 
     * @param varName variable name
     * @param nnz number of non zeroes
     * @return matrix object
     * @throws DMLRuntimeException if DMLRuntimeException occurs
     */
    public Pair<MatrixObject, Boolean> getSparseMatrixOutputForGPUInstruction(String varName, long nnz)
        throws DMLRuntimeException
    {
        MatrixObject mo = allocateGPUMatrixObject(varName);
        mo.getMatrixCharacteristics().setNonZeros(nnz);
				boolean allocated = mo.getGPUObject(getGPUContext(0)).acquireDeviceModifySparse();
        return new Pair<MatrixObject, Boolean>(mo, allocated);
    } 

	/**
	 * Allocates the {@link GPUObject} for a given LOPS Variable (eg. _mVar3)
	 * @param varName variable name
	 * @return matrix object
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public MatrixObject allocateGPUMatrixObject(String varName) throws DMLRuntimeException {
		MatrixObject mo = getMatrixObject(varName);
		if( mo.getGPUObject(getGPUContext(0)) == null ) {
			GPUObject newGObj = getGPUContext(0).createGPUObject(mo);
			// The lock is added here for an output block
			// so that any block currently in use is not deallocated by eviction on the GPU
			newGObj.addLock();
			mo.setGPUObject(getGPUContext(0), newGObj);
		}
		return mo;
	}

	public Pair<MatrixObject, Boolean> getMatrixInputForGPUInstruction(String varName)
			throws DMLRuntimeException 
	{
		GPUContext gCtx = getGPUContext(0);
		boolean copied = false;
		MatrixObject mo = getMatrixObject(varName);
		if(mo == null) {
			throw new DMLRuntimeException("No matrix object available for variable:" + varName);
		}

		boolean acquired = false;
		if( mo.getGPUObject(gCtx) == null ) {
			GPUObject newGObj = gCtx.createGPUObject(mo);
			mo.setGPUObject(gCtx, newGObj);
		} else if( !mo.getGPUObject(gCtx).isInputAllocated() ) {
			mo.acquireRead();
			acquired = true;
		}

		copied = mo.getGPUObject(gCtx).acquireDeviceRead();
		if(acquired) {
			mo.release();
		}
		return new Pair<MatrixObject, Boolean>(mo, copied);
	}
	
	/**
	 * Unpins a currently pinned matrix variable. 
	 * 
	 * @param varName variable name
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
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
		mo.getGPUObject(getGPUContext(0)).releaseInput();
	}
	
	/**
	 * Pins a frame variable into memory and returns the internal frame block.
	 * 
	 * @param varName variable name
	 * @return frame block
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
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
	 * @param varName variable name
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void releaseFrameInput(String varName) 
		throws DMLRuntimeException 
	{
		FrameObject fo = getFrameObject(varName);
		fo.release();
	}
	
	public ScalarObject getScalarInput(CPOperand input) throws DMLRuntimeException {
		return getScalarInput(input.getName(), input.getValueType(), input.isLiteral());
	}
	
	public ScalarObject getScalarInput(String name, ValueType vt, boolean isLiteral)
		throws DMLRuntimeException 
	{
		if ( isLiteral ) {
			return ScalarObjectFactory.createScalarObject(vt, name);
		}
		else {
			Data obj = getVariable(name);
			if (obj == null)
				throw new DMLRuntimeException("Unknown variable: " + name);
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
		if(mo.getGPUObject(getGPUContext(0)) == null || !mo.getGPUObject(getGPUContext(0)).isAllocated()) {
			throw new DMLRuntimeException("No output is allocated on GPU");
		}
		mo.getGPUObject(getGPUContext(0)).releaseOutput();
	}

	public void setMatrixOutput(String varName, MatrixBlock outputData) 
			throws DMLRuntimeException 
	{
		MatrixObject mo = getMatrixObject(varName);
		mo.acquireModify(outputData);
	    mo.release();
	    setVariable(varName, mo);
	}

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

	public void setFrameOutput(String varName, FrameBlock outputData) 
		throws DMLRuntimeException 
	{
		FrameObject fo = getFrameObject(varName);
		if( outputData.getNumColumns()>0 && outputData.getSchema()!=null )
			fo.setValueType(outputData.getSchema()[0]);
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
	 * 
	 * @param varList variable list
	 * @return map of old cleanup state of matrix objects
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
	 * 
	 * @param varList variable list
	 * @param varsState variable state
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
	 * @return variable list as strings
	 */
	public ArrayList<String> getVarList()
	{
		ArrayList<String> varlist = new ArrayList<String>();
		varlist.addAll(_variables.keySet());	
		return varlist;
	}

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

	public void updateDebugState( int index ) throws DMLRuntimeException 
	{
		if(DMLScript.ENABLE_DEBUG_MODE) {
			_dbState.getPC().setProgramBlockNumber(index);
		}
	}

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
	 * 
	 * @param currInst current instruction
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
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
