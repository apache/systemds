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

package org.apache.sysds.runtime.controlprogram.context;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.yarn.webapp.hamlet2.HamletSpec;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysds.runtime.controlprogram.caching.TensorObject;
import org.apache.sysds.runtime.controlprogram.caching.prescientbuffer.IOTrace;
import org.apache.sysds.runtime.controlprogram.federated.MatrixLineagePair;
import org.apache.sysds.runtime.controlprogram.paramserv.homomorphicEncryption.SEALClient;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObjectFactory;
import org.apache.sysds.runtime.instructions.gpu.context.CSRPointer;
import org.apache.sysds.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysds.runtime.instructions.gpu.context.GPUObject;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;
import org.apache.sysds.runtime.lineage.LineageDebugger;
import org.apache.sysds.runtime.lineage.LineageGPUCacheEviction;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaData;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.utils.Statistics;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Future;
import java.util.stream.Collectors;
import java.util.Queue;

public class ExecutionContext {
	protected static final Log LOG = LogFactory.getLog(ExecutionContext.class.getName());

	//program reference (e.g., function repository)
	protected Program _prog = null;
	
	//symbol table
	protected LocalVariableMap _variables;
	protected long _tid = -1;
	protected boolean _autoCreateVars;

	//lineage map, cache, prepared dedup blocks
	protected Lineage _lineage;

	protected SEALClient _seal_client;

	//parfor temporary functions (created by eval)
	protected Set<String> _fnNames;

	private IOTrace _ioTrace;

	public IOTrace getIOTrace() {
		if (_ioTrace == null) {
			_ioTrace = new IOTrace();
		}
		return _ioTrace;
	}

	public void setIOTrace(IOTrace ioTrace) {
		_ioTrace = ioTrace;
	}

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
		_autoCreateVars = false;
		_lineage = allocateLineage ? new Lineage() : null;
		_prog = prog;
		_fnNames = new HashSet<>();
	}

	public ExecutionContext(LocalVariableMap vars) {
		_variables = vars;
		_autoCreateVars = false;
		_lineage = null;
		_prog = null;
		_fnNames = new HashSet<>();
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

	public boolean isAutoCreateVars() {
		return _autoCreateVars;
	}

	public void setAutoCreateVars(boolean flag) {
		_autoCreateVars = flag;
	}

	public void setTID(long tid) {
		_tid = tid;
	}

	public long getTID() {
		return _tid;
	}

	public void setSealClient(SEALClient seal_client) {
		_seal_client = seal_client;
	}

	public SEALClient getSealClient() {
		return _seal_client;
	}

	/**
	 *
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
		// Set the single-GPU context in the lineage cache
		if (!LineageCacheConfig.ReuseCacheType.isNone())
			LineageGPUCacheEviction.setGPUContext(gpuContexts.get(0));
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
	
	public boolean containsVariable(CPOperand operand) {
		return containsVariable(operand.getName());
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
		Data tmp = _variables.get(varname);
		if( tmp == null )
			throw new DMLRuntimeException(getNonExistingVarError(varname));
		return tmp.getMetaData();
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
			throw new DMLRuntimeException(getNonExistingVarError(varname));
		if( !(dat instanceof MatrixObject) )
			throw new DMLRuntimeException("Variable '"+varname+"' is not a matrix: "+dat.getClass().getName());
		
		return (MatrixObject) dat;
	}

	public MatrixLineagePair getMatrixLineagePair(CPOperand cpo) {
		return getMatrixLineagePair(cpo.getName());
	}

	public MatrixLineagePair getMatrixLineagePair(String varname) {
		MatrixObject mo = getMatrixObject(varname);
		if(mo == null)
			return null;
		return MatrixLineagePair.of(mo, DMLScript.LINEAGE ? getLineageItem(varname) : null);
	}

	public TensorObject getTensorObject(String varname) {
		Data dat = getVariable(varname);

		//error handling if non existing or no matrix
		if( dat == null )
			throw new DMLRuntimeException(getNonExistingVarError(varname));
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
			throw new DMLRuntimeException(getNonExistingVarError(varname));
		if( !(dat instanceof FrameObject) )
			throw new DMLRuntimeException("Variable '"+varname+"' is not a frame: "+dat.getDataType());
		return (FrameObject) dat;
	}

	public CacheableData<?> getCacheableData(CPOperand input) {
		return getCacheableData(input.getName());
	}
	
	public CacheableData<?> getCacheableData(String varname) {
		Data dat = getVariable(varname);
		//error handling if non existing or no matrix
		if( dat == null )
			throw new DMLRuntimeException(getNonExistingVarError(varname));
		if( !(dat instanceof CacheableData<?>) )
			throw new DMLRuntimeException("Variable '"+varname+"' is not a matrix, tensor or frame.");
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
	
	public MatrixBlock getMatrixInput(CPOperand input) {
		return getMatrixObject(input.getName()).acquireRead();
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
		MatrixCharacteristics mc = new MatrixCharacteristics(nrows, ncols, mo.getBlocksize());
		mo.setMetaData(new MetaDataFormat(mc, ((MetaDataFormat)oldMetaData).getFileFormat()));
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
		return 	getDenseMatrixOutputForGPUInstruction(varName, numRows, numCols, true);
	}

	public Pair<MatrixObject, Boolean> getDenseMatrixOutputForGPUInstruction(String varName, long numRows, long numCols,
		boolean initialize)
	{
		MatrixObject mo = allocateGPUMatrixObject(varName, numRows, numCols);
		boolean allocated = mo.getGPUObject(getGPUContext(0)).acquireDeviceModifyDense(initialize);
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
		return getSparseMatrixOutputForGPUInstruction(varName, numRows, numCols, nnz, true);
	}

	public Pair<MatrixObject, Boolean> getSparseMatrixOutputForGPUInstruction(String varName, long numRows, long numCols,
		long nnz, boolean initialize)
	{
		MatrixObject mo = allocateGPUMatrixObject(varName, numRows, numCols);
		mo.getDataCharacteristics().setNonZeros(nnz);
		boolean allocated = mo.getGPUObject(getGPUContext(0)).acquireDeviceModifySparse(initialize);
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

		try {
			dim1 = validateDimensions(mo.getNumRows(), numRows);
			dim2 = validateDimensions(mo.getNumColumns(), numCols);
		}
		catch(DMLRuntimeException e) {
			throw new DMLRuntimeException("Incorrect dimensions given to allocateGPUMatrixObject: [" + numRows + "," +
					numCols + "], " + "[" + mo.getNumRows() + "," + mo.getNumColumns() + "]", e);
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
	
	public long getGPUDensePointerAddress(MatrixObject obj) {
		if(obj.getGPUObject(getGPUContext(0)) == null)
				return 0;
			else
				return obj.getGPUObject(getGPUContext(0)).getDensePointerAddress();
	}
	
	public CSRPointer getGPUSparsePointerAddress(MatrixObject obj) {
		if(obj.getGPUObject(getGPUContext(0)) == null)
			throw new RuntimeException("No CSRPointer for MatrixObject " + obj.toString());
		else
			return obj.getGPUObject(getGPUContext(0)).getJcudaSparseMatrixPtr();
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
	
	public FrameBlock getFrameInput(CPOperand input) {
		return getFrameInput(input.getName());
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
	
	public void releaseFrameInputs(CPOperand[] inputs) {
		Arrays.stream(inputs).filter(in -> in.isFrame())
			.forEach(in -> releaseFrameInput(in.getName()));
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

	public ListObject getListObject(CPOperand input) {
		return getListObject(input.getName());
	}
	
	public ListObject getListObject(String name) {
		Data dat = getVariable(name);
		//error handling if non existing or no list
		if (dat == null)
			throw new DMLRuntimeException(getNonExistingVarError(name));
		if (!(dat instanceof ListObject))
			throw new DMLRuntimeException("Variable '" + name + "' is not a list.");
		return (ListObject) dat;
	}
	
	private List<MatrixObject> getMatricesFromList(ListObject lo) {
		List<MatrixObject> ret = new ArrayList<>();
		for (Data e : lo.getData()) {
			if (e instanceof MatrixObject)
				ret.add((MatrixObject)e);
			else if (e instanceof ListObject)
				ret.addAll(getMatricesFromList((ListObject)e));
			else
				throw new DMLRuntimeException("List must contain only matrices or lists for rbind/cbind.");
		}
		return ret;
	}

	public void releaseMatrixOutputForGPUInstruction(String varName) {
		MatrixObject mo = getMatrixObject(varName);
		if(mo.getGPUObject(getGPUContext(0)) == null || !mo.getGPUObject(getGPUContext(0)).isAllocated()) {
			throw new DMLRuntimeException("No output is allocated on GPU");
		}
		setMetaData(varName, new MetaDataFormat(mo.getDataCharacteristics(), FileFormat.BINARY));
		mo.getGPUObject(getGPUContext(0)).releaseOutput();
	}
	
	public void setMatrixOutput(String varName, MatrixBlock outputData) {
		setMatrixOutputAndLineage(varName, outputData, null);
	}

	public void setMatrixOutputAndLineage(CPOperand var, MatrixBlock outputData, LineageItem li) {
		setMatrixOutputAndLineage(var.getName(), outputData, li);
	}
	
	public void setMatrixOutputAndLineage(String varName, MatrixBlock outputData, LineageItem li) {
		if( isAutoCreateVars() && !containsVariable(varName) )
			setVariable(varName, createMatrixObject(outputData));
		MatrixObject mo = getMatrixObject(varName);
		mo.acquireModify(outputData);
		mo.setCacheLineage(li);
		mo.release();
	}

	public void setMatrixOutputAndLineage(String varName, Future<MatrixBlock> fmb, LineageItem li) {
		if (isAutoCreateVars() && !containsVariable(varName)) {
			setVariable(varName, new MatrixObjectFuture(Types.ValueType.FP64,
				OptimizerUtils.getUniqueTempFileName(), fmb));
		}
		MatrixObject mo = getMatrixObject(varName);
		MatrixObjectFuture fmo = new MatrixObjectFuture(mo, fmb);
		fmo.setCacheLineage(li);
		setVariable(varName, fmo);
	}

	public void setMatrixOutput(String varName, Future<MatrixBlock> fmb) {
		setMatrixOutputAndLineage(varName, fmb, null);
	}

	public void setMatrixOutput(String varName, MatrixBlock outputData, UpdateType flag) {
		if( isAutoCreateVars() && !containsVariable(varName) )
			setVariable(varName, createMatrixObject(outputData));
		if( flag.isInPlace() ) {
			//modify metadata to carry update status
			MatrixObject mo = getMatrixObject(varName);
			mo.setUpdateType( flag );
		}
		setMatrixOutput(varName, outputData);
	}

	public void setTensorOutput(String varName, TensorBlock outputData) {
		TensorObject to = getTensorObject(varName);
		to.acquireModify(outputData);
		to.release();
		setVariable(varName, to);
	}
	
	public void setFrameOutput(String varName, FrameBlock outputData) {
		if( isAutoCreateVars() && !containsVariable(varName) )
			setVariable(varName, createFrameObject(outputData));
		FrameObject fo = getFrameObject(varName);
		fo.setSchema(outputData.getSchema());
		fo.acquireModify(outputData);
		fo.release();
		setVariable(varName, fo);
	}

	public static CacheableData<?> createCacheableData(CacheBlock<?> cb) {
		if( cb instanceof MatrixBlock )
			return createMatrixObject((MatrixBlock) cb);
		else if( cb instanceof FrameBlock )
			return createFrameObject((FrameBlock) cb);
		return null;
	}

	public static MatrixObject createMatrixObject(MatrixBlock mb) {
		final long nRow = mb.getNumRows(), nCol = mb.getNumColumns();
		final int bz = ConfigurationManager.getBlocksize();
		MetaData md = new MetaDataFormat(new MatrixCharacteristics(nRow, nCol, bz), FileFormat.BINARY);
		return new MatrixObject(Types.ValueType.FP64, OptimizerUtils.getUniqueTempFileName(), md, mb);
	}

	public static MatrixObject createMatrixObject(DataCharacteristics dc) {
		final long nRow = dc.getRows(), nCol = dc.getCols();
		final int bz =  dc.getBlocksize() == -1 ? ConfigurationManager.getBlocksize() : dc.getBlocksize();
		MetaData md = new MetaDataFormat(new MatrixCharacteristics(nRow, nCol, bz), FileFormat.BINARY);
		return new MatrixObject(Types.ValueType.FP64, OptimizerUtils.getUniqueTempFileName(), md);
	}

	public static FrameObject createFrameObject(DataCharacteristics dc) {
		FrameObject ret = new FrameObject(OptimizerUtils.getUniqueTempFileName());
		ret.setMetaData(new MetaDataFormat(new MatrixCharacteristics(
			dc.getRows(), dc.getCols()), FileFormat.BINARY));
		ret.getMetaData().getDataCharacteristics()
			.setBlocksize(ConfigurationManager.getBlocksize());
		return ret;
	}

	public static FrameObject createFrameObject(FrameBlock fb) {
		FrameObject ret = new FrameObject(OptimizerUtils.getUniqueTempFileName());
		ret.acquireModify(fb);
		ret.setMetaData(new MetaDataFormat(new MatrixCharacteristics(
			fb.getNumRows(), fb.getNumColumns()), FileFormat.BINARY));
		ret.release();
		return ret;
	}

	public List<MatrixBlock> getMatrixInputs(CPOperand[] inputs) {
		return getMatrixInputs(inputs, false);
	}
	
	public List<MatrixBlock> getMatrixInputs(CPOperand[] inputs, boolean includeList) {
		List<MatrixBlock> ret = Arrays.stream(inputs).filter(in -> in.isMatrix())
			.map(in -> getMatrixInput(in.getName())).collect(Collectors.toList());
		
		if (includeList) {
			List<ListObject> lolist = Arrays.stream(inputs).filter(in -> in.isList())
				.map(in -> getListObject(in.getName())).collect(Collectors.toList());
			for (ListObject lo : lolist)
				ret.addAll( getMatricesFromList(lo).stream()
					.map(mo -> mo.acquireRead()).collect(Collectors.toList()));
		}
		
		return ret;
	}
	
	public List<ScalarObject> getScalarInputs(CPOperand[] inputs) {
		return Arrays.stream(inputs).filter(in -> in.isScalar())
			.map(in -> getScalarInput(in)).collect(Collectors.toList());
	}
	
	public List<FrameBlock> getFrameInputs(CPOperand[] inputs) {
		return Arrays.stream(inputs).filter(in -> in.isFrame())
			.map(in -> getFrameInput(in)).collect(Collectors.toList());
	}
	
	public void releaseMatrixInputs(CPOperand[] inputs) {
		releaseMatrixInputs(inputs, false);
	}

	public void releaseMatrixInputs(CPOperand[] inputs, boolean includeList) {
		Arrays.stream(inputs).filter(in -> in.isMatrix())
			.forEach(in -> releaseMatrixInput(in.getName()));

		if (includeList) {
			List<ListObject> lolist = Arrays.stream(inputs).filter(in -> in.isList())
				.map(in -> getListObject(in.getName())).collect(Collectors.toList());
			for (ListObject lo : lolist)
				getMatricesFromList(lo).stream().forEach(mo -> mo.release());
		}
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
	public Queue<Boolean> pinVariables(List<String> varList)
	{
		// step 1) get current cleanupFlag status information
		Queue<Boolean> varsStates = new LinkedList<>();
		for (String varName : varList) {
			Data dat = _variables.get(varName);
			if (dat instanceof CacheableData<?>)
				varsStates.add(((CacheableData<?>)dat).isCleanupEnabled());
			else if (dat instanceof ListObject)
				varsStates.addAll(((ListObject)dat).getCleanupStates());
		}

		// step 2) pin variables
		for (String varName : varList) {
			Data dat = _variables.get(varName);
			if (dat instanceof CacheableData<?>)
				((CacheableData<?>)dat).enableCleanup(false);
			else if (dat instanceof ListObject)
				((ListObject)dat).enableCleanup(false);
		}
		
		return varsStates;
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
	public void unpinVariables(List<String> varList, Queue<Boolean> varsState) {
		for (String varName : varList) {
			Data dat = _variables.get(varName);
			if (dat instanceof CacheableData<?>)
				((CacheableData<?>)dat).enableCleanup(varsState.poll());
			else if (dat instanceof ListObject)
				((ListObject)dat).enableCleanup(varsState);
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
				cleanupDataObject(dat2);
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
				mo.clearData(getTID()); //clean cached data
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
	
	public boolean isFederated(CPOperand input) {
		Data data = getVariable(input);
		if(data instanceof CacheableData && ((CacheableData<?>) data).isFederated())
			return true;
		return false;
	}
	
	public boolean isFederated(CPOperand input, FType type) {
		Data data = getVariable(input);
		if(data instanceof CacheableData && ((CacheableData<?>) data).isFederated(type))
			return true;
		return false;
	}

	public void traceLineage(Instruction inst) {
		if( _lineage == null )
			throw new DMLRuntimeException("Lineage Trace unavailable.");
		// TODO bra: store all newly created lis in active list
		_lineage.trace(inst, this);
	}
	
	public void maintainLineageDebuggerInfo(Instruction inst) {
		if( _lineage == null )
			throw new DMLRuntimeException("Lineage Trace unavailable.");
		LineageDebugger.maintainSpecialValueBits(_lineage, inst, this);
	}

	public LineageItem getLineageItem(CPOperand input) {
		return getLineageItem(input.getName());
	}

	public LineageItem getLineageItem(String varname) {
		if( _lineage == null )
			throw new DMLRuntimeException("Lineage Trace unavailable.");
		return _lineage.get(varname);
	}

	public LineageItem getOrCreateLineageItem(CPOperand input) {
		if( _lineage == null )
			throw new DMLRuntimeException("Lineage Trace unavailable.");
		return _lineage.getOrCreate(input);
	}

	public void replaceLineageItem(String varname, LineageItem li) {
		if (!LineageCacheConfig.isLineageTraceReuse())
			return;
		if( _lineage == null )
			throw new DMLRuntimeException("Lineage Trace unavailable.");
		if (_lineage.get(varname) == null)
			throw new DMLRuntimeException("Lineage item does not exist for "+varname);
		//Passed lineage trace should be equivalent to the live lineage trace
		//corresponding to varname. Replacing reduces memory and probing overheads.
		_lineage.set(varname, li);
	}
	
	private static String getNonExistingVarError(String varname) {
		return "Variable '" + varname + "' does not exist in the symbol table.";
	}

	public void addTmpParforFunction(String fname) {
		_fnNames.add(fname);
	}

	public Set<String> getTmpParforFunctions() {
		return _fnNames;
	}

	@Override
	public String toString(){
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName().toString());
		if(_prog != null)
			sb.append("\nProgram: " + _prog.toString());
		if(_variables != null)
			sb.append("\nLocalVariableMap: " + _variables.toString());
		if(_lineage != null)
			sb.append("\nLineage: " + _lineage.toString());
		return sb.toString();
	}
}
