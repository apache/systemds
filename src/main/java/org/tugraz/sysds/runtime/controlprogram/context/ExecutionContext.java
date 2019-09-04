/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.runtime.controlprogram.context;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.LocalVariableMap;
import org.tugraz.sysds.runtime.controlprogram.Program;
import org.tugraz.sysds.runtime.controlprogram.caching.CacheableData;
import org.tugraz.sysds.runtime.controlprogram.caching.FrameObject;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.tugraz.sysds.runtime.controlprogram.caching.TensorObject;
import org.tugraz.sysds.runtime.data.TensorBlock;
import org.tugraz.sysds.runtime.instructions.Instruction;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;
import org.tugraz.sysds.runtime.instructions.cp.Data;
import org.tugraz.sysds.runtime.instructions.cp.ListObject;
import org.tugraz.sysds.runtime.instructions.cp.ScalarObject;
import org.tugraz.sysds.runtime.instructions.cp.ScalarObjectFactory;
import org.tugraz.sysds.runtime.instructions.gpu.context.GPUContext;
import org.tugraz.sysds.runtime.instructions.gpu.context.GPUObject;
import org.tugraz.sysds.runtime.lineage.Lineage;
import org.tugraz.sysds.runtime.lineage.LineageItem;
import org.tugraz.sysds.runtime.lineage.LineagePath;
import org.tugraz.sysds.runtime.matrix.data.FrameBlock;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.matrix.data.Pair;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.meta.MetaData;
import org.tugraz.sysds.runtime.meta.MetaDataFormat;
import org.tugraz.sysds.runtime.util.HDFSTool;
import org.tugraz.sysds.utils.Statistics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;


public class ExecutionContext {
	protected static final Log LOG = LogFactory.getLog(ExecutionContext.class.getName());

	//program reference (e.g., function repository)
	protected Program _prog = null;
	
	//symbol table
	protected LocalVariableMap _variables;

	//lineage map, cache, prepared dedup blocks
	protected Lineage _lineage;
	protected LineagePath _lineagePath = new LineagePath();

	/**
	 * List of {@link GPUContext}s owned by this {@link ExecutionContext}
	 */
	protected List<GPUContext> _gpuContexts = new ArrayList<>();
	
	protected ExecutionContext() {
		//protected constructor to force use of ExecutionContextFactory
		this( true, DMLScript.LINEAGE, null );
	}

	protected ExecutionContext( boolean allocateVariableMap, boolean allocateLineage, Program prog ) {
		//protected constructor to force use of ExecutionContextFactory
		_variables = allocateVariableMap ? new LocalVariableMap() : null;
		_lineage = allocateLineage ? new Lineage() : null;
		_prog = prog;
	}

	public Program getProgram(){
		return _prog;
	}

	public void setProgram(Program prog) {
		_prog = prog;
	}
	
	public LocalVariableMap getVariables() {
		return _variables;
	}
	
	public void setVariables(LocalVariableMap vars) {
		_variables = vars;
	}

	public Lineage getLineage() {
		return _lineage;
	}

	public void setLineage(Lineage lineage) {
		_lineage = lineage;
	}

	public LineagePath getLineagePath(){
		return _lineagePath;
	}

	public void setLineagePath(LineagePath lp){
		_lineagePath = lp;
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
	
	public Data getVariable(CPOperand operand) {
		return operand.getDataType().isScalar() ?
			getScalarInput(operand) : getVariable(operand.getName());
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

	public void setMetaData(String fname, MetaData md) {
		_variables.get(fname).setMetaData(md);
	}
	
	public MetaData getMetaData(String varname) {
		return _variables.get(varname).getMetaData();
	}
	
	public boolean isMatrixObject(String varname) {
		Data dat = getVariable(varname);
		return (dat!= null && dat instanceof MatrixObject);
	}
	
	public MatrixObject getMatrixObject(CPOperand input) {
		return getMatrixObject(input.getName());
	}

	public MatrixObject getMatrixObject(String varname) {
		Data dat = getVariable(varname);
		
		//error handling if non existing or no matrix
		if( dat == null )
			throw new DMLRuntimeException("Variable '"+varname+"' does not exist in the symbol table.");
		if( !(dat instanceof MatrixObject) )
			throw new DMLRuntimeException("Variable '"+varname+"' is not a matrix.");
		
		return (MatrixObject) dat;
	}

	public TensorObject getTensorObject(String varname) {
		Data dat = getVariable(varname);

		//error handling if non existing or no matrix
		if( dat == null )
			throw new DMLRuntimeException("Variable '"+varname+"' does not exist in the symbol table.");
		if( !(dat instanceof TensorObject) )
			throw new DMLRuntimeException("Variable '"+varname+"' is not a tensor.");

		return (TensorObject) dat;
	}

	public boolean isFrameObject(String varname) {
		Data dat = getVariable(varname);
		return (dat!= null && dat instanceof FrameObject);
	}
	
	public FrameObject getFrameObject(CPOperand input) {
		return getFrameObject(input.getName());
	}
	
	public FrameObject getFrameObject(String varname) {
		Data dat = getVariable(varname);
		//error handling if non existing or no matrix
		if( dat == null )
			throw new DMLRuntimeException("Variable '"+varname+"' does not exist in the symbol table.");
		if( !(dat instanceof FrameObject) )
			throw new DMLRuntimeException("Variable '"+varname+"' is not a frame.");
		return (FrameObject) dat;
	}

	public CacheableData<?> getCacheableData(CPOperand input) {
		return getCacheableData(input.getName());
	}
	
	public CacheableData<?> getCacheableData(String varname) {
		Data dat = getVariable(varname);
		//error handling if non existing or no matrix
		if( dat == null )
			throw new DMLRuntimeException("Variable '"+varname+"' does not exist in the symbol table.");
		if( !(dat instanceof CacheableData<?>) )
			throw new DMLRuntimeException("Variable '"+varname+"' is not a matrix or frame.");
		return (CacheableData<?>) dat;
	}

	public void releaseCacheableData(String varname) {
		getCacheableData(varname).release();
	}

	public DataCharacteristics getDataCharacteristics(String varname) {
		return getMetaData(varname).getDataCharacteristics();
	}
	
	/**
	 * Pins a matrix variable into memory and returns the internal matrix block.
	 * 
	 * @param varName variable name
	 * @return matrix block
	 */
	public MatrixBlock getMatrixInput(String varName) {
		return getMatrixObject(varName).acquireRead();
	}

	/**
	 * Pins a matrix variable into memory and returns the internal matrix block.
	 *
	 * @param varName variable name
	 * @return matrix block
	 */
	public TensorBlock getTensorInput(String varName) {
		return getTensorObject(varName).acquireRead();
	}

	public void setMetaData(String varName, long nrows, long ncols) {
		MatrixObject mo = getMatrixObject(varName);
		if(mo.getNumRows() == nrows && mo.getNumColumns() == ncols) 
			return;
		MetaData oldMetaData = mo.getMetaData();
		if( oldMetaData == null || !(oldMetaData instanceof MetaDataFormat) )
			throw new DMLRuntimeException("Metadata not available");
		MatrixCharacteristics mc = new MatrixCharacteristics(nrows, ncols,
			(int) mo.getBlocksize());
		mo.setMetaData(new MetaDataFormat(mc, 
			((MetaDataFormat)oldMetaData).getOutputInfo(),
			((MetaDataFormat)oldMetaData).getInputInfo()));
	}
	
	/**
	 * Compares two potential dimensions d1 and d2 and return the one which is not -1.
	 * This method is useful when the dimensions are not known at compile time, but are known at runtime.
	 *  
	 * @param d1 dimension1
	 * @param d2 dimension1
	 * @return valid d1 or d2
	 */
	private static long validateDimensions(long d1, long d2) {
		if(d1 >= 0 && d2 >= 0 && d1 != d2) {
			throw new DMLRuntimeException("Incorrect dimensions:" + d1 + " != " + d2);
		}
		return Math.max(d1, d2);
	}

	/**
	 * Allocates a dense matrix on the GPU (for output)
	 * @param varName	name of the output matrix (known by this {@link ExecutionContext})
	 * @param numRows number of rows of matrix object
	 * @param numCols number of columns of matrix object
	 * @return a pair containing the wrapping {@link MatrixObject} and a boolean indicating whether a cuda memory allocation took place (as opposed to the space already being allocated)
	 */
	public Pair<MatrixObject, Boolean> getDenseMatrixOutputForGPUInstruction(String varName, long numRows, long numCols) {
		MatrixObject mo = allocateGPUMatrixObject(varName, numRows, numCols);
		boolean allocated = mo.getGPUObject(getGPUContext(0)).acquireDeviceModifyDense();
		mo.getDataCharacteristics().setNonZeros(-1);
		return new Pair<>(mo, allocated);
	}

	/**
	 * Allocates a sparse matrix in CSR format on the GPU.
	 * Assumes that mat.getNumRows() returns a valid number
	 * 
	 * @param varName variable name
	 * @param numRows number of rows of matrix object
	 * @param numCols number of columns of matrix object
	 * @param nnz number of non zeroes
	 * @return matrix object
	 */
	public Pair<MatrixObject, Boolean> getSparseMatrixOutputForGPUInstruction(String varName, long numRows, long numCols, long nnz) {
		MatrixObject mo = allocateGPUMatrixObject(varName, numRows, numCols);
		mo.getDataCharacteristics().setNonZeros(nnz);
				boolean allocated = mo.getGPUObject(getGPUContext(0)).acquireDeviceModifySparse();
		return new Pair<>(mo, allocated);
	}

	/**
	 * Allocates the {@link GPUObject} for a given LOPS Variable (eg. _mVar3)
	 * @param varName variable name
	 * @param numRows number of rows of matrix object
	 * @param numCols number of columns of matrix object
	 * @return matrix object
	 */
	public MatrixObject allocateGPUMatrixObject(String varName, long numRows, long numCols) {
		MatrixObject mo = getMatrixObject(varName);
		long dim1 = -1; long dim2 = -1;
		DMLRuntimeException e = null;
		try {
			dim1 = validateDimensions(mo.getNumRows(), numRows);
		} catch(DMLRuntimeException e1) {
			e = e1;
		}
		try {
			dim2 = validateDimensions(mo.getNumColumns(), numCols);
		} catch(DMLRuntimeException e1) {
			e = e1;
		}
		if(e != null) {
			throw new DMLRuntimeException("Incorrect dimensions given to allocateGPUMatrixObject: [" + numRows + "," + numCols + "], "
					+ "[" + mo.getNumRows() + "," + mo.getNumColumns() + "]", e);
		}
		if(dim1 != mo.getNumRows() || dim2 != mo.getNumColumns()) {
			// Set unknown dimensions
			mo.getDataCharacteristics().setDimension(dim1, dim2);
		}
		if( mo.getGPUObject(getGPUContext(0)) == null ) {
			GPUObject newGObj = getGPUContext(0).createGPUObject(mo);
			mo.setGPUObject(getGPUContext(0), newGObj);
		}
		// The lock is added here for an output block
		// so that any block currently in use is not deallocated by eviction on the GPU
		mo.getGPUObject(getGPUContext(0)).addWriteLock();
		return mo;
	}

	public MatrixObject getMatrixInputForGPUInstruction(String varName, String opcode) {
		GPUContext gCtx = getGPUContext(0);
		MatrixObject mo = getMatrixObject(varName);
		if(mo == null) {
			throw new DMLRuntimeException("No matrix object available for variable:" + varName);
		}

		if( mo.getGPUObject(gCtx) == null ) {
			GPUObject newGObj = gCtx.createGPUObject(mo);
			mo.setGPUObject(gCtx, newGObj);
		}
		// No need to perform acquireRead here because it is performed in copyFromHostToDevice
		mo.getGPUObject(gCtx).acquireDeviceRead(opcode);
		return mo;
	}
	
	/**
	 * Unpins a currently pinned matrix variable and update fine-grained statistics. 
	 * 
	 * @param varName variable name
	 */
	public void releaseMatrixInput(String varName) {
		getMatrixObject(varName).release();
	}
	
	public void releaseMatrixInput(String... varNames) {
		for( String varName : varNames )
			releaseMatrixInput(varName);
	}
	
	public void releaseMatrixInputForGPUInstruction(String varName) {
		getMatrixObject(varName).getGPUObject(getGPUContext(0)).releaseInput();
	}
	
	/**
	 * Pins a frame variable into memory and returns the internal frame block.
	 * 
	 * @param varName variable name
	 * @return frame block
	 */
	public FrameBlock getFrameInput(String varName) {
		return getFrameObject(varName).acquireRead();
	}
	
	/**
	 * Unpins a currently pinned frame variable. 
	 * 
	 * @param varName variable name
	 */
	public void releaseFrameInput(String varName) {
		getFrameObject(varName).release();
	}

	public void releaseTensorInput(String varName) {
		getTensorObject(varName).release();
	}

	public void releaseTensorInput(String... varNames) {
		for( String varName : varNames )
			releaseTensorInput(varName);
	}

	public ScalarObject getScalarInput(CPOperand input) {
		return input.isLiteral() ? input.getLiteral() : 
			getScalarInput(input.getName(), input.getValueType(), false);
	}
	
	public ScalarObject getScalarInput(String name, ValueType vt, boolean isLiteral) {
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

	public void setScalarOutput(String varName, ScalarObject so) {
		setVariable(varName, so);
	}

	public ListObject getListObject(String name) {
		Data dat = getVariable(name);
		//error handling if non existing or no list
		if (dat == null)
			throw new DMLRuntimeException("Variable '" + name + "' does not exist in the symbol table.");
		if (!(dat instanceof ListObject))
			throw new DMLRuntimeException("Variable '" + name + "' is not a list.");
		return (ListObject) dat;
	}

	public void releaseMatrixOutputForGPUInstruction(String varName) {
		MatrixObject mo = getMatrixObject(varName);
		if(mo.getGPUObject(getGPUContext(0)) == null || !mo.getGPUObject(getGPUContext(0)).isAllocated()) {
			throw new DMLRuntimeException("No output is allocated on GPU");
		}
		setMetaData(varName, new MetaDataFormat(mo.getDataCharacteristics(), OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
		mo.getGPUObject(getGPUContext(0)).releaseOutput();
	}
	
	public void setMatrixOutput(String varName, MatrixBlock outputData) {
		MatrixObject mo = getMatrixObject(varName);
		mo.acquireModify(outputData);
		mo.release();
		setVariable(varName, mo);
	}

	public void setMatrixOutput(String varName, MatrixBlock outputData, UpdateType flag) {
		if( flag.isInPlace() ) {
			//modify metadata to carry update status
			MatrixObject mo = getMatrixObject(varName);
			mo.setUpdateType( flag );
		}
		setMatrixOutput(varName, outputData);
	}

	public void setMatrixOutput(String varName, MatrixBlock outputData, UpdateType flag, String opcode) {
		setMatrixOutput(varName, outputData, flag);
	}

	public void setTensorOutput(String varName, TensorBlock outputData) {
		TensorObject to = getTensorObject(varName);
		to.acquireModify(outputData);
		to.release();
		setVariable(varName, to);
	}

	public void setFrameOutput(String varName, FrameBlock outputData) {
		FrameObject fo = getFrameObject(varName);
		fo.acquireModify(outputData);
		fo.release();
		setVariable(varName, fo);
	}
	
	public List<MatrixBlock> getMatrixInputs(CPOperand[] inputs) {
		return Arrays.stream(inputs).filter(in -> in.isMatrix())
			.map(in -> getMatrixInput(in.getName())).collect(Collectors.toList());
	}
	
	public List<ScalarObject> getScalarInputs(CPOperand[] inputs) {
		return Arrays.stream(inputs).filter(in -> in.isScalar())
			.map(in -> getScalarInput(in)).collect(Collectors.toList());
	}
	
	public void releaseMatrixInputs(CPOperand[] inputs) {
		Arrays.stream(inputs).filter(in -> in.isMatrix())
			.forEach(in -> releaseMatrixInput(in.getName()));
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
	 * @return indicator vector of old cleanup state of matrix objects
	 */
	public boolean[] pinVariables(List<String> varList) 
	{
		//analyze list variables
		int nlist = 0;
		int nlistItems = 0;
		for( int i=0; i<varList.size(); i++ ) {
			Data dat = _variables.get(varList.get(i));
			if( dat instanceof ListObject ) {
				nlistItems += ((ListObject)dat).getNumCacheableData();
				nlist++;
			}
		}
		
		//2-pass approach since multiple vars might refer to same matrix object
		boolean[] varsState = new boolean[varList.size()-nlist+nlistItems];
		
		//step 1) get current information
		for( int i=0, pos=0; i<varList.size(); i++ ) {
			Data dat = _variables.get(varList.get(i));
			if( dat instanceof CacheableData<?>  )
				varsState[pos++] = ((CacheableData<?>)dat).isCleanupEnabled();
			else if( dat instanceof ListObject )
				for( Data dat2 : ((ListObject)dat).getData() )
					if( dat2 instanceof CacheableData<?> )
						varsState[pos++] = ((CacheableData<?>)dat2).isCleanupEnabled();
		}
		
		//step 2) pin variables
		for( int i=0; i<varList.size(); i++ ) {
			Data dat = _variables.get(varList.get(i));
			if( dat instanceof CacheableData<?> )
				((CacheableData<?>)dat).enableCleanup(false);
			else if( dat instanceof ListObject )
				for( Data dat2 : ((ListObject)dat).getData() )
					if( dat2 instanceof CacheableData<?> )
						((CacheableData<?>)dat2).enableCleanup(false);
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
	public void unpinVariables(List<String> varList, boolean[] varsState) {
		for( int i=0, pos=0; i<varList.size(); i++ ) {
			Data dat = _variables.get(varList.get(i));
			if( dat instanceof CacheableData<?> )
				((CacheableData<?>)dat).enableCleanup(varsState[pos++]);
			else if( dat instanceof ListObject )
				for( Data dat2 : ((ListObject)dat).getData() )
					if( dat2 instanceof CacheableData<?> )
						((CacheableData<?>)dat2).enableCleanup(varsState[pos++]);
		}
	}
	
	/**
	 * NOTE: No order guaranteed, so keep same list for pin and unpin. 
	 * 
	 * @return list of all variable names.
	 */
	public ArrayList<String> getVarList() {
		return new ArrayList<>(_variables.keySet());
	}
	
	/**
	 * NOTE: No order guaranteed, so keep same list for pin and unpin. 
	 * 
	 * @return list of all variable names of partitioned matrices.
	 */
	public ArrayList<String> getVarListPartitioned() {
		ArrayList<String> ret = new ArrayList<>();
		for( String var : _variables.keySet() ) {
			Data dat = _variables.get(var);
			if( dat instanceof MatrixObject 
				&& ((MatrixObject)dat).isPartitioned() )
				ret.add(var);
		}
		return ret;
	}
	
	public final void cleanupDataObject(Data dat) {
		if( dat == null ) return;
		if ( dat instanceof CacheableData )
			cleanupCacheableData( (CacheableData<?>)dat );
		else if( dat instanceof ListObject )
			for( Data dat2 : ((ListObject)dat).getData() )
				if( dat2 instanceof CacheableData<?> )
					cleanupCacheableData( (CacheableData<?>)dat2 );
	}
	
	public void cleanupCacheableData(CacheableData<?> mo) {
		if (DMLScript.JMLC_MEM_STATISTICS)
			Statistics.removeCPMemObject(System.identityHashCode(mo));
		//early abort w/o scan of symbol table if no cleanup required
		boolean fileExists = (mo.isHDFSFileExists() && mo.getFileName() != null);
		if( !CacheableData.isCachingActive() && !fileExists )
			return;
		
		try {
			//compute ref count only if matrix cleanup actually necessary
			if ( mo.isCleanupEnabled() && !getVariables().hasReferences(mo) )  {
				mo.clearData(); //clean cached data
				if( fileExists ) {
					HDFSTool.deleteFileIfExistOnHDFS(mo.getFileName());
					HDFSTool.deleteFileIfExistOnHDFS(mo.getFileName()+".mtd");
				}
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	public void traceLineage(Instruction inst) {
		if( _lineage == null )
			throw new DMLRuntimeException("Lineage Trace unavailable.");
		_lineage.trace(inst, this);
	}

	public LineageItem getLineageItem(CPOperand input) {
		if( _lineage == null )
			throw new DMLRuntimeException("Lineage Trace unavailable.");
		return _lineage.get(input);
	}

	public LineageItem getOrCreateLineageItem(CPOperand input) {
		if( _lineage == null )
			throw new DMLRuntimeException("Lineage Trace unavailable.");
		return _lineage.getOrCreate(input);
	}
}
