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

package org.tugraz.sysds.hops;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.hops.recompile.Recompiler;
import org.tugraz.sysds.hops.recompile.Recompiler.ResetType;
import org.tugraz.sysds.lops.Binary;
import org.tugraz.sysds.lops.BinaryScalar;
import org.tugraz.sysds.lops.CSVReBlock;
import org.tugraz.sysds.lops.Checkpoint;
import org.tugraz.sysds.lops.Data;
import org.tugraz.sysds.lops.Lop;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.lops.LopsException;
import org.tugraz.sysds.lops.Nary;
import org.tugraz.sysds.lops.ParameterizedBuiltin;
import org.tugraz.sysds.lops.ReBlock;
import org.tugraz.sysds.lops.Ternary;
import org.tugraz.sysds.lops.Unary;
import org.tugraz.sysds.lops.UnaryCP;
import org.tugraz.sysds.parser.ParseInfo;
import org.tugraz.sysds.runtime.controlprogram.LocalVariableMap;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.tugraz.sysds.runtime.instructions.gpu.context.GPUContextPool;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.util.UtilFunctions;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;


public abstract class Hop implements ParseInfo
{
	protected static final Log LOG =  LogFactory.getLog(Hop.class.getName());
	
	public static final long CPThreshold = 2000;

	// static variable to assign an unique ID to every hop that is created
	private static IDSequence _seqHopID = new IDSequence();
	
	protected final long _ID;
	protected String _name;
	protected DataType _dataType;
	protected ValueType _valueType;
	protected boolean _visited = false;
	protected long _dim1 = -1;
	protected long _dim2 = -1;
	protected int _blocksize = -1;
	protected long _nnz = -1;
	protected UpdateType _updateType = UpdateType.COPY;

	protected ArrayList<Hop> _parent = new ArrayList<>();
	protected ArrayList<Hop> _input = new ArrayList<>();

	protected ExecType _etype = null; //currently used exec type
	protected ExecType _etypeForced = null; //exec type forced via platform or external optimizer
	
	// Estimated size for the output produced from this Hop
	protected double _outputMemEstimate = OptimizerUtils.INVALID_SIZE;
	
	// Estimated size for the entire operation represented by this Hop
	// It includes the memory required for all inputs as well as the output 
	protected double _memEstimate = OptimizerUtils.INVALID_SIZE;
	protected double _processingMemEstimate = 0;
	protected double _spBroadcastMemEstimate = 0;
	
	// indicates if there are unknowns during compilation 
	// (in that case re-complication ensures robustness and efficiency)
	protected boolean _requiresRecompile = false;
	
	// indicates if the output of this hop needs to be reblocked
	// (usually this happens on persistent reads dataops)
	protected boolean _requiresReblock = false;
	
	// indicates if the output of this hop needs to be checkpointed (cached)
	// (the default storage level for caching is not yet exposed here)
	protected boolean _requiresCheckpoint = false;
	
	// indicates if the output of this hops needs to contain materialized empty blocks 
	// if those exists; otherwise only blocks w/ non-zero values are materialized
	protected boolean _outputEmptyBlocks = true;
	
	private Lop _lops = null;
	
	protected Hop(){
		//default constructor for clone
		_ID = getNextHopID();
	}
		
	public Hop(String l, DataType dt, ValueType vt) {
		this();
		setName(l);
		setDataType(dt);
		setValueType(vt);
	}

	
	private static long getNextHopID() {
		return _seqHopID.getNextID();
	}
	
	public long getHopID() {
		return _ID;
	}

	/**
	 * Check whether this Hop has a correct number of inputs.
	 *
	 * (Some Hops can have a variable number of inputs, such as DataOp, DataGenOp, ParameterizedBuiltinOp,
	 * ReorgOp, TernaryOp, QuaternaryOp, MultipleOp, DnnOp, and SpoofFusedOp.)
	 *
	 * Parameterized Hops (such as DataOp) can check that the number of parameters matches the number of inputs.
	 *
	 */
	public abstract void checkArity();
	
	public ExecType getExecType()
	{
		return _etype;
	}
	
	public void resetExecType()
	{
		_etype = null;
	}

	public ExecType getForcedExecType()
	{
		return _etypeForced;
	}

	public void setForcedExecType(ExecType etype)
	{
		_etypeForced = etype;
	}

	public abstract boolean allowsAllExecTypes();
	
	/**
	 * Defines if this operation is transpose-safe, which means that
	 * the result of op(input) is equivalent to op(t(input)).
	 * Usually, this applies to aggregate operations with fixed output
	 * dimension. Finally, this information is very useful in order to
	 * safely optimize the plan for sparse vectors, which otherwise
	 * would be (currently) always represented dense.
	 * 
	 * 
	 * @return always returns false
	 */
	public boolean isTransposeSafe()
	{
		//by default: its conservatively define as unsafe
		return false;
	}
	
	public void checkAndSetForcedPlatform()
	{
		if(DMLScript.USE_ACCELERATOR && DMLScript.FORCE_ACCELERATOR && isGPUEnabled())
			_etypeForced = ExecType.GPU; // enabled with -gpu force option
		else if ( DMLScript.getGlobalExecMode() == ExecMode.SINGLE_NODE ) {
			if(OptimizerUtils.isMemoryBasedOptLevel() && DMLScript.USE_ACCELERATOR && isGPUEnabled()) {
				// enabled with -exec singlenode -gpu option
				_etypeForced = findExecTypeByMemEstimate();
				if(_etypeForced != ExecType.CP && _etypeForced != ExecType.GPU)
					_etypeForced = ExecType.CP;
			}
			else {
				// enabled with -exec singlenode option
				_etypeForced = ExecType.CP;  
			}
		}
		else if ( DMLScript.getGlobalExecMode() == ExecMode.SPARK )
			_etypeForced = ExecType.SPARK; // enabled with -exec spark option
	}
	
	public void checkAndSetInvalidCPDimsAndSize()
	{
		if( _etype == ExecType.CP || _etype == ExecType.GPU ) {
			//check dimensions of output and all inputs (INTEGER)
			boolean invalid = !hasValidCPDimsAndSize();
			
			//force exec type mr if necessary
			if( invalid ) { 
				if( DMLScript.getGlobalExecMode() == ExecMode.HYBRID )
					_etype = ExecType.SPARK;
			}
		}
	}
	
	public boolean hasValidCPDimsAndSize() {
		boolean invalid = !OptimizerUtils.isValidCPDimensions(_dim1, _dim2);
		for( Hop in : getInput() )
			invalid |= !OptimizerUtils.isValidCPDimensions(in._dim1, in._dim2);
		return !invalid;
	}

	public boolean hasMatrixInputWithDifferentBlocksizes() {
		for( Hop c : getInput() ) {
			if( c.getDataType()==DataType.MATRIX
				&& getBlocksize() != c.getBlocksize() )
			{
				return true;
			}
		}
		
		return false;
	}
	
	public void setRequiresReblock(boolean flag) {
		_requiresReblock = flag;
	}
	
	public boolean requiresReblock() {
		return _requiresReblock;
	}
	
	public void setRequiresCheckpoint(boolean flag) {
		_requiresCheckpoint = flag;
	}
	
	public boolean requiresCheckpoint() {
		return _requiresCheckpoint;
	}
	
	public void constructAndSetLopsDataFlowProperties() {
		//Step 1: construct reblock lop if required (output of hop)
		constructAndSetReblockLopIfRequired();
		
		//Step 3: construct checkpoint lop if required (output of hop or reblock)
		constructAndSetCheckpointLopIfRequired();
	}

	private void constructAndSetReblockLopIfRequired() 
	{
		//determine execution type
		ExecType et = ExecType.CP;
		if( DMLScript.getGlobalExecMode() != ExecMode.SINGLE_NODE 
			&& !(getDataType()==DataType.SCALAR) )
		{
			et = ExecType.SPARK;
		}

		//add reblock lop to output if required
		if( _requiresReblock && et != ExecType.CP )
		{
			Lop input = getLops();
			Lop reblock = null;
			
			try
			{
				if( this instanceof DataOp  // CSV
					&& ((DataOp)this).getDataOpType() == DataOpTypes.PERSISTENTREAD
					&& ((DataOp)this).getInputFormatType() == FileFormatTypes.CSV  )
				{
					reblock = new CSVReBlock( input, getBlocksize(), 
						getDataType(), getValueType(), et);
				}
				else //TEXT / MM / BINARYBLOCK / BINARYCELL
				{
					reblock = new ReBlock( input, getBlocksize(), 
						getDataType(), getValueType(), _outputEmptyBlocks, et);
				}
			}
			catch( LopsException ex ) {
				throw new HopsException(ex);
			}
		
			setOutputDimensions( reblock );
			setLineNumbers( reblock );
			setLops( reblock );
		}
	}

	private void constructAndSetCheckpointLopIfRequired() {
		//determine execution type
		ExecType et = ExecType.CP;
		if( OptimizerUtils.isSparkExecutionMode() && getDataType()!=DataType.SCALAR ) {
			//conditional checkpoint based on memory estimate 
			et = ( Recompiler.checkCPCheckpoint(getDataCharacteristics() )
				|| _etypeForced == ExecType.CP ) ? ExecType.CP : ExecType.SPARK;
		}

		//add checkpoint lop to output if required
		if( _requiresCheckpoint && et != ExecType.CP )
		{
			try
			{
				//investigate need for serialized storage of large sparse matrices
				//(compile- instead of runtime-level for better debugging)
				boolean serializedStorage = false;
				if( getDataType()==DataType.MATRIX && dimsKnown(true) ) {
					double matrixPSize = OptimizerUtils.estimatePartitionedSizeExactSparsity(_dim1, _dim2, _blocksize, _nnz);
					double dataCache = SparkExecutionContext.getDataMemoryBudget(true, true);
					serializedStorage = MatrixBlock.evalSparseFormatInMemory(_dim1, _dim2, _nnz)
						&& matrixPSize > dataCache //sparse in-memory does not fit in agg mem 
						&& (OptimizerUtils.getSparsity(_dim1, _dim2, _nnz) < MatrixBlock.ULTRA_SPARSITY_TURN_POINT
							|| !Checkpoint.CHECKPOINT_SPARSE_CSR ); //ultra-sparse or sparse w/o csr
				}
				else if( !dimsKnown(true) ) {
					setRequiresRecompile();
				}
			
				//construct checkpoint w/ right storage level
				Lop input = getLops();
				Lop chkpoint = new Checkpoint(input, getDataType(), getValueType(), 
						serializedStorage ? Checkpoint.getSerializeStorageLevelString() :
						Checkpoint.getDefaultStorageLevelString() );
				
				setOutputDimensions( chkpoint );
				setLineNumbers( chkpoint );
				setLops( chkpoint );
			}
			catch( LopsException ex ) {
				throw new HopsException(ex);
			}
		}	
	}

	public static Lop createOffsetLop( Hop hop, boolean repCols ) 
	{
		Lop offset = null;
		
		if( ConfigurationManager.isDynamicRecompilation() && hop.dimsKnown() )
		{
			// If dynamic recompilation is enabled and dims are known, we can replace the ncol with 
			// a literal in order to increase the piggybacking potential. This is safe because append 
			// is always marked for recompilation and hence, we have propagated the exact dimensions.
			offset = Data.createLiteralLop(ValueType.INT64, String.valueOf(repCols ? hop.getDim2() : hop.getDim1()));
		}
		else
		{
			offset = new UnaryCP(hop.constructLops(), 
				repCols ? UnaryCP.OperationTypes.NCOL : UnaryCP.OperationTypes.NROW, 
				DataType.SCALAR, ValueType.INT64);
		}
		
		offset.getOutputParameters().setDimensions(0, 0, 0, -1);
		offset.setAllPositions(hop.getFilename(), hop.getBeginLine(), hop.getBeginColumn(), hop.getEndLine(), hop.getEndColumn());
		
		return offset;
	}
	
	public void setOutputEmptyBlocks(boolean flag) {
		_outputEmptyBlocks = flag;
	}
	
	public boolean isOutputEmptyBlocks() {
		return _outputEmptyBlocks;
	}
	

	protected double getInputOutputSize() {
		return _outputMemEstimate
			+ _processingMemEstimate
			+ getInputSize();
	}
	
	public double getInputOutputSize(Collection<String> exclVars) {
		return _outputMemEstimate
			+ _processingMemEstimate
			+ getInputSize(exclVars);
	}
	
	/**
	 * Returns the memory estimate for the output produced from this Hop.
	 * It must be invoked only within Hops. From outside Hops, one must 
	 * only use getMemEstimate(), which gives memory required to store 
	 * all inputs and the output.
	 * 
	 * @return output size memory estimate
	 */
	protected double getOutputSize() {
		return _outputMemEstimate;
	}
	
	protected double getInputSize() {
		return getInputSize(null);
	}

	protected double getInputSize(Collection<String> exclVars) {
		double sum = 0;
		int len = _input.size();
		for( int i=0; i<len; i++ ) { //for all inputs
			Hop hi = _input.get(i);
			if( exclVars != null && exclVars.contains(hi.getName()) )
				continue;
			double hmout = hi.getOutputMemEstimate();
			if( hmout > 1024*1024 ) {//for relevant sizes
				//check if already included in estimate (if an input is used
				//multiple times it is still only required once in memory)
				//(not that this check benefits from common subexpression elimination)
				boolean flag = false;
				for( int j=0; j<i; j++ )
					flag |= (hi == _input.get(j));
				hmout = flag ? 0 : hmout;
			}
			sum += hmout;
		}
		
		return sum;
	}

	protected double getInputSize( int pos ){
		double ret = 0;
		if( _input.size()>pos )
			ret = _input.get(pos)._outputMemEstimate;
		return ret;
	}
	
	protected double getIntermediateSize() {
		return _processingMemEstimate;
	}
	
	/**
	 * NOTES:
	 * * Purpose: Whenever the output dimensions / sparsity of a hop are unknown, this hop
	 *   should store its worst-case output statistics (if known) in that table. Subsequent
	 *   hops can then
	 * * Invocation: Intended to be called for ALL root nodes of one Hops DAG with the same
	 *   (initially empty) memo table.
	 * 
	 * @return memory estimate
	 */
	public double getMemEstimate()
	{
		if ( OptimizerUtils.isMemoryBasedOptLevel() ) {
			if ( ! isMemEstimated() ) {
				//LOG.warn("Nonexisting memory estimate - reestimating w/o memo table.");
				computeMemEstimate( new MemoTable() ); 
			}
			return _memEstimate;
		}
		else {
			return OptimizerUtils.INVALID_SIZE;
		}
	}
	
	/**
	 * Sets memory estimate in bytes
	 * 
	 * @param mem memory estimate
	 */
	public void setMemEstimate( double mem )
	{
		_memEstimate = mem;
	}
	
	public void clearMemEstimate()
	{
		_memEstimate = OptimizerUtils.INVALID_SIZE;
	}

	public boolean isMemEstimated() 
	{
		return (_memEstimate != OptimizerUtils.INVALID_SIZE);
	}

	//wrappers for meaningful public names to memory estimates.
	
	public double getInputMemEstimate()
	{
		return getInputSize();
	}
	
	public double getOutputMemEstimate()
	{
		return getOutputSize();
	}

	public double getIntermediateMemEstimate()
	{
		return getIntermediateSize();
	}
	
	public double getSpBroadcastSize()
	{
		return _spBroadcastMemEstimate;
	}
	
	/**
	 * Computes the estimate of memory required to store the input/output of this hop in memory. 
	 * This is the default implementation (orchestration of hop-specific implementation) 
	 * that should suffice for most hops. If a hop requires more control, this method should
	 * be overwritten with awareness of (1) output estimates, and (2) propagation of worst-case
	 * matrix characteristics (dimensions, sparsity).  
	 * 
	 * TODO remove memo table and, on constructor refresh, inference in refresh, single compute mem,
	 * maybe general computeMemEstimate, flags to indicate if estimate or not. 
	 * 
	 * @param memo memory table
	 */
	public void computeMemEstimate( MemoTable memo )
	{
		long[] wstats = null; 
		
		////////
		//Step 1) Compute hop output memory estimate (incl size inference) 
		
		switch( getDataType() )
		{
			case SCALAR: {
				//memory estimate always known
				if( getValueType()== ValueType.FP64) //default case
					_outputMemEstimate = OptimizerUtils.DOUBLE_SIZE;
				else //literalops, dataops
					_outputMemEstimate = computeOutputMemEstimate( _dim1, _dim2, _nnz );
				break;
			}
			case FRAME:
			case MATRIX:
			case TENSOR:
			case LIST:
			{
				//1a) mem estimate based on exactly known dimensions and sparsity
				if( dimsKnown(true) ) { 
					//nnz always exactly known (see dimsKnown(true))
					_outputMemEstimate = computeOutputMemEstimate( _dim1, _dim2, _nnz );
				}
				//1b) infer output statistics and mem estimate based on worst-case statistics
				else if( memo.hasInputStatistics(this) )
				{
					//infer the output stats
					wstats = inferOutputCharacteristics(memo);
					
					if( wstats != null && wstats[0] >= 0 && wstats[1] >= 0 ) {
						//use worst case characteristics to estimate mem
						long lnnz = ((wstats[2]>=0)?wstats[2]:wstats[0]*wstats[1]);
						_outputMemEstimate = computeOutputMemEstimate( wstats[0], wstats[1], lnnz );
						
						//propagate worst-case estimate
						memo.memoizeStatistics(getHopID(), wstats[0], wstats[1], wstats[2]);
					}
					else if( dimsKnown() ) {
						//nnz unknown, estimate mem as dense
						long lnnz = _dim1*_dim2;
						_outputMemEstimate = computeOutputMemEstimate( _dim1, _dim2, lnnz );
					}
					else {
						//unknown output size
						_outputMemEstimate = OptimizerUtils.DEFAULT_SIZE;
					}
				}
				//1c) mem estimate based on exactly known dimensions and unknown sparsity
				//(required e.g., for datagenops w/o any input statistics)
				else if( dimsKnown() ) {
					//nnz unknown, estimate mem as dense
					long lnnz = _dim1*_dim2;
					_outputMemEstimate = computeOutputMemEstimate( _dim1, _dim2, lnnz );
				}
				//1d) fallback: unknown output size
				else {
					_outputMemEstimate = OptimizerUtils.DEFAULT_SIZE;
				}
				
				break;
			}
			case UNKNOWN: {
				//memory estimate always unknown
				_outputMemEstimate = OptimizerUtils.DEFAULT_SIZE;
				break;
			}
		}
		
		////////
		//Step 2) Compute hop intermediate memory estimate  
		
		//note: ensure consistency w/ step 1 (for simplified debugging)	
		
		if( dimsKnown(true) ) { //incl scalar output
			//nnz always exactly known (see dimsKnown(true))
			_processingMemEstimate = computeIntermediateMemEstimate( _dim1, _dim2, _nnz );
		}
		else if( wstats!=null ) {
			//use worst case characteristics to estimate mem
			long lnnz = ((wstats[2]>=0)?wstats[2]:wstats[0]*wstats[1]);
			_processingMemEstimate = computeIntermediateMemEstimate( wstats[0], wstats[1], lnnz );
		}
		else if( dimsKnown() ){
			//nnz unknown, estimate mem as dense
			long lnnz = _dim1 * _dim2;
			_processingMemEstimate = computeIntermediateMemEstimate(_dim1, _dim2, lnnz);
		}
		
		
		////////
		//Step 3) Compute final hop memory estimate  
		
		//final estimate (sum of inputs/intermediates/output)
		_memEstimate = getInputOutputSize();
	}
	
	/**
	 * Computes the output matrix characteristics (rows, cols, nnz) based on worst-case output
	 * and/or input estimates. Should return null if dimensions are unknown.
	 * 
	 * @param memo memory table
	 * @return output characteristics as a long array
	 */
	protected abstract long[] inferOutputCharacteristics( MemoTable memo );

	/**
	 * Recursively computes memory estimates for all the Hops in the DAG rooted at the 
	 * current hop pointed by <code>this</code>.
	 * 
	 * @param memo memory table
	 */
	public void refreshMemEstimates( MemoTable memo ) {
		if( isVisited() )
			return;
		for( Hop h : this.getInput() )
			h.refreshMemEstimates( memo );
		computeMemEstimate( memo );
		setVisited();
	}

	/**
	 * This method determines the execution type (CP, MR) based ONLY on the 
	 * estimated memory footprint required for this operation, which includes 
	 * memory for all inputs and the output represented by this Hop.
	 * 
	 * It is used when <code>OptimizationType = MEMORY_BASED</code>.
	 * This optimization schedules an operation to CP whenever inputs+output 
	 * fit in memory -- note that this decision MAY NOT be optimal in terms of 
	 * execution time.
	 * 
	 * @return execution type
	 */
	protected ExecType findExecTypeByMemEstimate() {
		ExecType et = null;
		char c = ' ';
		double memEst = getMemEstimate();
		if ( memEst < OptimizerUtils.getLocalMemBudget() ) {
			if (DMLScript.USE_ACCELERATOR && isGPUEnabled() && memEst < GPUContextPool.initialGPUMemBudget())
				et = ExecType.GPU;
			else
				et = ExecType.CP;
		}
		else {
			if( DMLScript.getGlobalExecMode() == ExecMode.HYBRID )
				et = ExecType.SPARK;
			
			c = '*';
		}

		if (LOG.isDebugEnabled()){
			String s = String.format("  %c %-5s %-8s (%s,%s)  %s", c, getHopID(), getOpString(), OptimizerUtils.toMB(_outputMemEstimate), OptimizerUtils.toMB(_memEstimate), et);
			//System.out.println(s);
			LOG.debug(s);
		}
		
		return et;
	}

	public ArrayList<Hop> getParent() {
		return _parent;
	}

	public ArrayList<Hop> getInput() {
		return _input;
	}
	
	public void addInput( Hop h ) {
		_input.add(h);
		h._parent.add(this);
	}
	
	public void addAllInputs( ArrayList<Hop> list ) {
		for( Hop h : list )
			addInput(h);
	}

	public int getBlocksize() {
		return _blocksize;
	}

	public void setBlocksize(int blen) {
		_blocksize = blen;
	}
	
	public void setNnz(long nnz){
		_nnz = nnz;
	}
	
	public long getNnz(){
		return _nnz;
	}

	public void setUpdateType(UpdateType update){
		_updateType = update;
	}
	
	public UpdateType getUpdateType(){
		return _updateType;
	}

	public abstract Lop constructLops();

	protected abstract ExecType optFindExecType();
	
	public abstract String getOpString();

	// ========================================================================================
	// Design doc: Memory estimation of GPU
	// 1. Since not all operator are supported on GPU, isGPUEnabled indicates whether an operation 
	// is enabled for GPU. This method doesnot take into account any memory estimates.
	// 2. To simplify memory estimation logic, the methods computeOutputMemEstimate and computeIntermediateMemEstimate
	// should return maximum of memory required for GPU and CP operators. 
	// 3. Additionally, these methods are guarded so that when -gpu flag is not provided, additional memory overhead due to GPU
	// are ignored. For example: sparse-to-dense conversion on GPU. 
	// 4. (WIP) Every GPU operators should respect the memory returned by computeIntermediateMemEstimate (and computeOutputMemEstimate - see below point).
	// 5. (WIP) Every GPU operator should create output in the same format as the corresponding CP operator. That is,  computeOutputMemEstimate
	// are consistent across both CP and GPU in terms of worst-case.
	// 6. The drawback of using maximum memory (mem = Math.max(mem_gpu, mem_gpu)) are:
	// - GPU operator is not selected when mem_gpu < total memory available on GPU < mem
	// - CP operator is not selected (i.e. distributed operator compiled) when mem_cpu < driver memory budget < mem
	
	/**
	 * In memory-based optimizer mode (see OptimizerUtils.isMemoryBasedOptLevel()), 
	 * the exectype is determined by checking this method as well as memory budget of this Hop. 
	 * Please see findExecTypeByMemEstimate for more detail. 
	 * 
	 * This method is necessary because not all operator are supported efficiently
	 * on GPU (for example: operations on frames and scalar as well as operations such as table). 
	 * 
	 * @return true if the Hop is eligible for GPU Exectype.
	 */
	public abstract boolean isGPUEnabled();
	
	/**
	 * Computes the hop-specific output memory estimate in bytes. Should be 0 if not
	 * applicable. 
	 * 
	 * @param dim1 dimension 1
	 * @param dim2 dimension 2
	 * @param nnz number of non-zeros
	 * @return memory estimate
	 */
	protected abstract double computeOutputMemEstimate( long dim1, long dim2, long nnz );

	/**
	 * Computes the hop-specific intermediate memory estimate in bytes. Should be 0 if not
	 * applicable.
	 * 
	 * @param dim1 dimension 1
	 * @param dim2 dimension 2
	 * @param nnz number of non-zeros
	 * @return memory estimate
	 */
	protected abstract double computeIntermediateMemEstimate( long dim1, long dim2, long nnz );
	
	// ========================================================================================

	
	protected boolean isVector() {
		return (dimsKnown() && (_dim1 == 1 || _dim2 == 1) );
	}
	
	protected boolean areDimsBelowThreshold() {
		return (dimsKnown() && _dim1 <= Hop.CPThreshold && _dim2 <= Hop.CPThreshold );
	}
	
	public boolean dimsKnown() {
		return ( _dataType == DataType.SCALAR 
			|| ((_dataType==DataType.MATRIX || _dataType==DataType.FRAME) 
				&& _dim1 >= 0 && _dim2 >= 0) );
	}
	
	public boolean dimsKnown(boolean includeNnz) {
		return rowsKnown() && colsKnown()
			&& (_dataType.isScalar() || ((includeNnz) ? _nnz>=0 : true));
	}

	public boolean dimsKnownAny() {
		return rowsKnown() || colsKnown();
	}
	
	public boolean rowsKnown() {
		return _dataType.isScalar() || _dim1 >= 0;
	}
	
	public boolean colsKnown() {
		return _dataType.isScalar() || _dim2 >= 0;
	}
	
	public static void resetVisitStatus( ArrayList<Hop> hops ) {
		if( hops != null )
			for( Hop hopRoot : hops )
				hopRoot.resetVisitStatus();
	}
	
	public static void resetVisitStatus( ArrayList<Hop> hops, boolean force ) {
		if( !force )
			resetVisitStatus(hops);
		else {
			HashSet<Long> memo = new HashSet<>();
			if( hops != null )
				for( Hop hopRoot : hops )
					hopRoot.resetVisitStatusForced(memo);
		}
	}
	
	public Hop resetVisitStatus()  {
		if( !isVisited() )
			return this;
		for( Hop h : getInput() )
			h.resetVisitStatus();
		setVisited(false);
		return this;
	}
	
	public void resetVisitStatusForced(HashSet<Long> memo) {
		if( memo.contains(getHopID()) )
			return;
		for( Hop h : getInput() )
			h.resetVisitStatusForced(memo);
		setVisited(false);
		memo.add(getHopID());
	}

	public static void resetRecompilationFlag( ArrayList<Hop> hops, ExecType et, ResetType reset )
	{
		resetVisitStatus( hops );
		for( Hop hopRoot : hops )
			hopRoot.resetRecompilationFlag( et, reset );
	}
	
	public static void resetRecompilationFlag( Hop hops, ExecType et, ResetType reset )
	{
		hops.resetVisitStatus();
		hops.resetRecompilationFlag( et, reset );
	}
	
	private void resetRecompilationFlag( ExecType et, ResetType reset ) 
	{
		if( isVisited() )
			return;
		
		//process child hops
		for (Hop h : getInput())
			h.resetRecompilationFlag( et, reset );
		
		//reset recompile flag
		if( (et == null || getExecType() == et || getExecType() == null)
			&& (reset==ResetType.RESET || (reset==ResetType.RESET_KNOWN_DIMS && dimsKnown()))
			&& !(_requiresCheckpoint && getLops() instanceof Checkpoint && !dimsKnown(true)) ) {
			_requiresRecompile = false;
		}
		
		setVisited();
	}

	public long getDim1() {
		return _dim1;
	}

	public void setDim1(long dim1) {
		_dim1 = dim1;
	}

	public long getDim2() {
		return _dim2;
	}

	public void setDim2(long dim2) {
		_dim2 = dim2;
	}
	
	public long getLength() {
		return _dim1 * _dim2;
	}
	
	public double getSparsity() {
		return OptimizerUtils.getSparsity(_dim1, _dim2, _nnz);
	}
	
	public DataCharacteristics getDataCharacteristics() {
		return new MatrixCharacteristics(
			_dim1, _dim2, _blocksize, _nnz);
	}
	
	protected void setOutputDimensions(Lop lop) {
		lop.getOutputParameters().setDimensions(
			getDim1(), getDim2(), getBlocksize(), getNnz(), getUpdateType());
	}
	
	public Lop getLops() {
		return _lops;
	}

	public void setLops(Lop lops) {
		_lops = lops;
	}

	public boolean isVisited() {
		return _visited;
	}

	public DataType getDataType() {
		return _dataType;
	}
	
	public void setDataType( DataType dt ) {
		_dataType = dt;
	}
	
	public boolean isScalar() {
		return _dataType.isScalar();
	}
	
	public boolean isMatrix() {
		return _dataType.isMatrix();
	}

	public void setVisited() {
		setVisited(true);
	}
	
	public void setVisited(boolean flag) {
		_visited = flag;
	}

	public void setName(String _name) {
		this._name = _name;
	}

	public String getName() {
		return _name;
	}

	public ValueType getValueType() {
		return _valueType;
	}
	
	public void setValueType(ValueType vt) {
		_valueType = vt;
	}

	public enum OpOp1 {
		NOT, ABS, SIN, COS, TAN, ASIN, ACOS, ATAN, SINH, COSH, TANH, SIGN, SQRT, LOG, EXP, 
		CAST_AS_SCALAR, CAST_AS_MATRIX, CAST_AS_FRAME, CAST_AS_DOUBLE, CAST_AS_INT, CAST_AS_BOOLEAN,
		PRINT, ASSERT, EIGEN, NROW, NCOL, LENGTH, ROUND, IQM, STOP, CEIL, FLOOR, MEDIAN, INVERSE, CHOLESKY,
		SVD, EXISTS, LINEAGE,
		//cumulative sums, products, extreme values
		CUMSUM, CUMPROD, CUMMIN, CUMMAX, CUMSUMPROD,
		//fused ML-specific operators for performance 
		SPROP, //sample proportion: P * (1 - P)
		SIGMOID, //sigmoid function: 1 / (1 + exp(-X))
		LOG_NZ, //sparse-safe log; ppred(X,0,"!=")*log(X)
	}

	// Operations that require two operands
	public enum OpOp2 {
		PLUS, MINUS, MULT, DIV, MODULUS, INTDIV, LESS, LESSEQUAL, GREATER, GREATEREQUAL, EQUAL, NOTEQUAL, 
		MIN, MAX, AND, OR, XOR, LOG, POW, PRINT, CONCAT, QUANTILE, INTERQUANTILE, IQM,
		MOMENT, COV, CBIND, RBIND, SOLVE, MEDIAN, INVALID,
		//fused ML-specific operators for performance
		MINUS_NZ, //sparse-safe minus: X-(mean*ppred(X,0,!=))
		LOG_NZ, //sparse-safe log; ppred(X,0,"!=")*log(X,0.5)
		MINUS1_MULT, //1-X*Y
		BITWAND, BITWOR, BITWXOR, BITWSHIFTL, BITWSHIFTR, //bitwise operations
	}

	// Operations that require 3 operands
	public enum OpOp3 {
		QUANTILE, INTERQUANTILE, CTABLE, MOMENT, COV, PLUS_MULT, MINUS_MULT, IFELSE
	}
	
	// Operations that require 4 operands
	public enum OpOp4 {
		WSLOSS, //weighted sloss mm
		WSIGMOID, //weighted sigmoid mm
		WDIVMM, //weighted divide mm
		WCEMM, //weighted cross entropy mm
		WUMM //weighted unary mm
	}
	
	// Operations that require a variable number of operands
	public enum OpOpN {
		PRINTF, CBIND, RBIND, MIN, MAX, EVAL, LIST
	}
	
	public enum AggOp {
		SUM, SUM_SQ, MIN, MAX, TRACE, PROD, MEAN, VAR, MAXINDEX, MININDEX
	}

	public enum ReOrgOp {
		TRANS, DIAG, RESHAPE, SORT, REV
		//Note: Diag types are invalid because for unknown sizes this would 
		//create incorrect plans (now we try to infer it for memory estimates
		//and rewrites but the final choice is made during runtime)
		//DIAG_V2M, DIAG_M2V, 
	}
	
	public enum OpOpDnn {
		MAX_POOL, MAX_POOL_BACKWARD, AVG_POOL, AVG_POOL_BACKWARD,
		CONV2D, CONV2D_BACKWARD_FILTER, CONV2D_BACKWARD_DATA,
		BIASADD, BIASMULT, BATCH_NORM2D_TEST, CHANNEL_SUMS,
		UPDATE_NESTEROV_X
	}
	
	public enum DataGenMethod {
		RAND, SEQ, SINIT, SAMPLE, INVALID, TIME
	}

	public enum ParamBuiltinOp {
		INVALID, CDF, INVCDF, GROUPEDAGG, RMEMPTY, REPLACE, REXPAND,
		LOWER_TRI, UPPER_TRI,
		TRANSFORMAPPLY, TRANSFORMDECODE, TRANSFORMCOLMAP, TRANSFORMMETA,
		TOSTRING, LIST, PARAMSERV
	}

	public enum FileFormatTypes {
		TEXT, BINARY, MM, CSV, LIBSVM
	}

	public enum DataOpTypes {
		PERSISTENTREAD, PERSISTENTWRITE, TRANSIENTREAD, TRANSIENTWRITE, FUNCTIONOUTPUT
	}

	public enum Direction {
		RowCol, Row, Col
	}

	protected static final HashMap<DataOpTypes, org.tugraz.sysds.lops.Data.OperationTypes> HopsData2Lops;
	static {
		HopsData2Lops = new HashMap<>();
		HopsData2Lops.put(DataOpTypes.PERSISTENTREAD, org.tugraz.sysds.lops.Data.OperationTypes.READ);
		HopsData2Lops.put(DataOpTypes.PERSISTENTWRITE, org.tugraz.sysds.lops.Data.OperationTypes.WRITE);
		HopsData2Lops.put(DataOpTypes.TRANSIENTWRITE, org.tugraz.sysds.lops.Data.OperationTypes.WRITE);
		HopsData2Lops.put(DataOpTypes.TRANSIENTREAD, org.tugraz.sysds.lops.Data.OperationTypes.READ);
	}

	protected static final HashMap<Hop.AggOp, org.tugraz.sysds.lops.Aggregate.OperationTypes> HopsAgg2Lops;
	static {
		HopsAgg2Lops = new HashMap<>();
		HopsAgg2Lops.put(AggOp.SUM, org.tugraz.sysds.lops.Aggregate.OperationTypes.KahanSum);
		HopsAgg2Lops.put(AggOp.SUM_SQ, org.tugraz.sysds.lops.Aggregate.OperationTypes.KahanSumSq);
		HopsAgg2Lops.put(AggOp.TRACE, org.tugraz.sysds.lops.Aggregate.OperationTypes.KahanTrace);
		HopsAgg2Lops.put(AggOp.MIN, org.tugraz.sysds.lops.Aggregate.OperationTypes.Min);
		HopsAgg2Lops.put(AggOp.MAX, org.tugraz.sysds.lops.Aggregate.OperationTypes.Max);
		HopsAgg2Lops.put(AggOp.MAXINDEX, org.tugraz.sysds.lops.Aggregate.OperationTypes.MaxIndex);
		HopsAgg2Lops.put(AggOp.MININDEX, org.tugraz.sysds.lops.Aggregate.OperationTypes.MinIndex);
		HopsAgg2Lops.put(AggOp.PROD, org.tugraz.sysds.lops.Aggregate.OperationTypes.Product);
		HopsAgg2Lops.put(AggOp.MEAN, org.tugraz.sysds.lops.Aggregate.OperationTypes.Mean);
		HopsAgg2Lops.put(AggOp.VAR, org.tugraz.sysds.lops.Aggregate.OperationTypes.Var);
	}

	protected static final HashMap<ReOrgOp, org.tugraz.sysds.lops.Transform.OperationTypes> HopsTransf2Lops;
	static {
		HopsTransf2Lops = new HashMap<>();
		HopsTransf2Lops.put(ReOrgOp.TRANS, org.tugraz.sysds.lops.Transform.OperationTypes.Transpose);
		HopsTransf2Lops.put(ReOrgOp.REV, org.tugraz.sysds.lops.Transform.OperationTypes.Rev);
		HopsTransf2Lops.put(ReOrgOp.DIAG, org.tugraz.sysds.lops.Transform.OperationTypes.Diag);
		HopsTransf2Lops.put(ReOrgOp.RESHAPE, org.tugraz.sysds.lops.Transform.OperationTypes.Reshape);
		HopsTransf2Lops.put(ReOrgOp.SORT, org.tugraz.sysds.lops.Transform.OperationTypes.Sort);

	}
	
	protected static final HashMap<OpOpDnn, org.tugraz.sysds.lops.DnnTransform.OperationTypes> HopsConv2Lops;
	static {
		HopsConv2Lops = new HashMap<>();
		HopsConv2Lops.put(OpOpDnn.MAX_POOL, org.tugraz.sysds.lops.DnnTransform.OperationTypes.MAX_POOL);
		HopsConv2Lops.put(OpOpDnn.MAX_POOL_BACKWARD, org.tugraz.sysds.lops.DnnTransform.OperationTypes.MAX_POOL_BACKWARD);
		HopsConv2Lops.put(OpOpDnn.AVG_POOL, org.tugraz.sysds.lops.DnnTransform.OperationTypes.AVG_POOL);
		HopsConv2Lops.put(OpOpDnn.AVG_POOL_BACKWARD, org.tugraz.sysds.lops.DnnTransform.OperationTypes.AVG_POOL_BACKWARD);
		HopsConv2Lops.put(OpOpDnn.CONV2D, org.tugraz.sysds.lops.DnnTransform.OperationTypes.CONV2D);
		HopsConv2Lops.put(OpOpDnn.BIASADD, org.tugraz.sysds.lops.DnnTransform.OperationTypes.BIAS_ADD);
		HopsConv2Lops.put(OpOpDnn.BIASMULT, org.tugraz.sysds.lops.DnnTransform.OperationTypes.BIAS_MULTIPLY);
		HopsConv2Lops.put(OpOpDnn.CONV2D_BACKWARD_FILTER, org.tugraz.sysds.lops.DnnTransform.OperationTypes.CONV2D_BACKWARD_FILTER);
		HopsConv2Lops.put(OpOpDnn.CONV2D_BACKWARD_DATA, org.tugraz.sysds.lops.DnnTransform.OperationTypes.CONV2D_BACKWARD_DATA);
		HopsConv2Lops.put(OpOpDnn.BATCH_NORM2D_TEST, org.tugraz.sysds.lops.DnnTransform.OperationTypes.BATCH_NORM2D_TEST);
		HopsConv2Lops.put(OpOpDnn.CHANNEL_SUMS, org.tugraz.sysds.lops.DnnTransform.OperationTypes.CHANNEL_SUMS);
		HopsConv2Lops.put(OpOpDnn.UPDATE_NESTEROV_X, org.tugraz.sysds.lops.DnnTransform.OperationTypes.UPDATE_NESTEROV_X);
	}

	protected static final HashMap<Hop.Direction, org.tugraz.sysds.lops.PartialAggregate.DirectionTypes> HopsDirection2Lops;
	static {
		HopsDirection2Lops = new HashMap<>();
		HopsDirection2Lops.put(Direction.RowCol, org.tugraz.sysds.lops.PartialAggregate.DirectionTypes.RowCol);
		HopsDirection2Lops.put(Direction.Col, org.tugraz.sysds.lops.PartialAggregate.DirectionTypes.Col);
		HopsDirection2Lops.put(Direction.Row, org.tugraz.sysds.lops.PartialAggregate.DirectionTypes.Row);

	}

	protected static final HashMap<Hop.OpOp2, Binary.OperationTypes> HopsOpOp2LopsB;
	static {
		HopsOpOp2LopsB = new HashMap<>();
		HopsOpOp2LopsB.put(OpOp2.PLUS, Binary.OperationTypes.ADD);
		HopsOpOp2LopsB.put(OpOp2.MINUS, Binary.OperationTypes.SUBTRACT);
		HopsOpOp2LopsB.put(OpOp2.MULT, Binary.OperationTypes.MULTIPLY);
		HopsOpOp2LopsB.put(OpOp2.DIV, Binary.OperationTypes.DIVIDE);
		HopsOpOp2LopsB.put(OpOp2.MODULUS, Binary.OperationTypes.MODULUS);
		HopsOpOp2LopsB.put(OpOp2.INTDIV, Binary.OperationTypes.INTDIV);
		HopsOpOp2LopsB.put(OpOp2.MINUS1_MULT, Binary.OperationTypes.MINUS1_MULTIPLY);
		HopsOpOp2LopsB.put(OpOp2.LESS, Binary.OperationTypes.LESS_THAN);
		HopsOpOp2LopsB.put(OpOp2.LESSEQUAL, Binary.OperationTypes.LESS_THAN_OR_EQUALS);
		HopsOpOp2LopsB.put(OpOp2.GREATER, Binary.OperationTypes.GREATER_THAN);
		HopsOpOp2LopsB.put(OpOp2.GREATEREQUAL, Binary.OperationTypes.GREATER_THAN_OR_EQUALS);
		HopsOpOp2LopsB.put(OpOp2.EQUAL, Binary.OperationTypes.EQUALS);
		HopsOpOp2LopsB.put(OpOp2.NOTEQUAL, Binary.OperationTypes.NOT_EQUALS);
		HopsOpOp2LopsB.put(OpOp2.MIN, Binary.OperationTypes.MIN);
		HopsOpOp2LopsB.put(OpOp2.MAX, Binary.OperationTypes.MAX);
		HopsOpOp2LopsB.put(OpOp2.AND, Binary.OperationTypes.AND);
		HopsOpOp2LopsB.put(OpOp2.XOR, Binary.OperationTypes.XOR);
		HopsOpOp2LopsB.put(OpOp2.OR, Binary.OperationTypes.OR);
		HopsOpOp2LopsB.put(OpOp2.SOLVE, Binary.OperationTypes.SOLVE);
		HopsOpOp2LopsB.put(OpOp2.POW, Binary.OperationTypes.POW);
		HopsOpOp2LopsB.put(OpOp2.LOG, Binary.OperationTypes.NOTSUPPORTED);
		HopsOpOp2LopsB.put(OpOp2.BITWAND, Binary.OperationTypes.BW_AND);
		HopsOpOp2LopsB.put(OpOp2.BITWOR, Binary.OperationTypes.BW_OR);
		HopsOpOp2LopsB.put(OpOp2.BITWXOR, Binary.OperationTypes.BW_XOR);
		HopsOpOp2LopsB.put(OpOp2.BITWSHIFTL, Binary.OperationTypes.BW_SHIFTL);
		HopsOpOp2LopsB.put(OpOp2.BITWSHIFTR, Binary.OperationTypes.BW_SHIFTR);
	}

	protected static final HashMap<Hop.OpOp2, BinaryScalar.OperationTypes> HopsOpOp2LopsBS;
	static {
		HopsOpOp2LopsBS = new HashMap<>();
		HopsOpOp2LopsBS.put(OpOp2.PLUS, BinaryScalar.OperationTypes.ADD);
		HopsOpOp2LopsBS.put(OpOp2.MINUS, BinaryScalar.OperationTypes.SUBTRACT);
		HopsOpOp2LopsBS.put(OpOp2.MULT, BinaryScalar.OperationTypes.MULTIPLY);
		HopsOpOp2LopsBS.put(OpOp2.DIV, BinaryScalar.OperationTypes.DIVIDE);
		HopsOpOp2LopsBS.put(OpOp2.MODULUS, BinaryScalar.OperationTypes.MODULUS);
		HopsOpOp2LopsBS.put(OpOp2.INTDIV, BinaryScalar.OperationTypes.INTDIV);
		HopsOpOp2LopsBS.put(OpOp2.LESS, BinaryScalar.OperationTypes.LESS_THAN);
		HopsOpOp2LopsBS.put(OpOp2.LESSEQUAL, BinaryScalar.OperationTypes.LESS_THAN_OR_EQUALS);
		HopsOpOp2LopsBS.put(OpOp2.GREATER, BinaryScalar.OperationTypes.GREATER_THAN);
		HopsOpOp2LopsBS.put(OpOp2.GREATEREQUAL, BinaryScalar.OperationTypes.GREATER_THAN_OR_EQUALS);
		HopsOpOp2LopsBS.put(OpOp2.EQUAL, BinaryScalar.OperationTypes.EQUALS);
		HopsOpOp2LopsBS.put(OpOp2.NOTEQUAL, BinaryScalar.OperationTypes.NOT_EQUALS);
		HopsOpOp2LopsBS.put(OpOp2.MIN, BinaryScalar.OperationTypes.MIN);
		HopsOpOp2LopsBS.put(OpOp2.MAX, BinaryScalar.OperationTypes.MAX);
		HopsOpOp2LopsBS.put(OpOp2.AND, BinaryScalar.OperationTypes.AND);
		HopsOpOp2LopsBS.put(OpOp2.OR, BinaryScalar.OperationTypes.OR);
		HopsOpOp2LopsBS.put(OpOp2.XOR, BinaryScalar.OperationTypes.XOR);
		HopsOpOp2LopsBS.put(OpOp2.LOG, BinaryScalar.OperationTypes.LOG);
		HopsOpOp2LopsBS.put(OpOp2.POW, BinaryScalar.OperationTypes.POW);
		HopsOpOp2LopsBS.put(OpOp2.PRINT, BinaryScalar.OperationTypes.PRINT);
		HopsOpOp2LopsBS.put(OpOp2.BITWAND, BinaryScalar.OperationTypes.BW_AND);
		HopsOpOp2LopsBS.put(OpOp2.BITWOR, BinaryScalar.OperationTypes.BW_OR);
		HopsOpOp2LopsBS.put(OpOp2.BITWXOR, BinaryScalar.OperationTypes.BW_XOR);
		HopsOpOp2LopsBS.put(OpOp2.BITWSHIFTL, BinaryScalar.OperationTypes.BW_SHIFTL);
		HopsOpOp2LopsBS.put(OpOp2.BITWSHIFTR, BinaryScalar.OperationTypes.BW_SHIFTR);
	}

	protected static final HashMap<Hop.OpOp2, org.tugraz.sysds.lops.Unary.OperationTypes> HopsOpOp2LopsU;
	static {
		HopsOpOp2LopsU = new HashMap<>();
		HopsOpOp2LopsU.put(OpOp2.PLUS, org.tugraz.sysds.lops.Unary.OperationTypes.ADD);
		HopsOpOp2LopsU.put(OpOp2.MINUS, org.tugraz.sysds.lops.Unary.OperationTypes.SUBTRACT);
		HopsOpOp2LopsU.put(OpOp2.MULT, org.tugraz.sysds.lops.Unary.OperationTypes.MULTIPLY);
		HopsOpOp2LopsU.put(OpOp2.DIV, org.tugraz.sysds.lops.Unary.OperationTypes.DIVIDE);
		HopsOpOp2LopsU.put(OpOp2.MODULUS, org.tugraz.sysds.lops.Unary.OperationTypes.MODULUS);
		HopsOpOp2LopsU.put(OpOp2.INTDIV, org.tugraz.sysds.lops.Unary.OperationTypes.INTDIV);
		HopsOpOp2LopsU.put(OpOp2.MINUS1_MULT, org.tugraz.sysds.lops.Unary.OperationTypes.MINUS1_MULTIPLY);
		HopsOpOp2LopsU.put(OpOp2.LESSEQUAL, org.tugraz.sysds.lops.Unary.OperationTypes.LESS_THAN_OR_EQUALS);
		HopsOpOp2LopsU.put(OpOp2.LESS, org.tugraz.sysds.lops.Unary.OperationTypes.LESS_THAN);
		HopsOpOp2LopsU.put(OpOp2.GREATEREQUAL, org.tugraz.sysds.lops.Unary.OperationTypes.GREATER_THAN_OR_EQUALS);
		HopsOpOp2LopsU.put(OpOp2.GREATER, org.tugraz.sysds.lops.Unary.OperationTypes.GREATER_THAN);
		HopsOpOp2LopsU.put(OpOp2.EQUAL, org.tugraz.sysds.lops.Unary.OperationTypes.EQUALS);
		HopsOpOp2LopsU.put(OpOp2.NOTEQUAL, org.tugraz.sysds.lops.Unary.OperationTypes.NOT_EQUALS);
		HopsOpOp2LopsU.put(OpOp2.AND, org.tugraz.sysds.lops.Unary.OperationTypes.AND);
		HopsOpOp2LopsU.put(OpOp2.OR, org.tugraz.sysds.lops.Unary.OperationTypes.OR);
		HopsOpOp2LopsU.put(OpOp2.XOR, org.tugraz.sysds.lops.Unary.OperationTypes.XOR);
		HopsOpOp2LopsU.put(OpOp2.MAX, org.tugraz.sysds.lops.Unary.OperationTypes.MAX);
		HopsOpOp2LopsU.put(OpOp2.MIN, org.tugraz.sysds.lops.Unary.OperationTypes.MIN);
		HopsOpOp2LopsU.put(OpOp2.LOG, org.tugraz.sysds.lops.Unary.OperationTypes.LOG);
		HopsOpOp2LopsU.put(OpOp2.POW, org.tugraz.sysds.lops.Unary.OperationTypes.POW);
		HopsOpOp2LopsU.put(OpOp2.MINUS_NZ, org.tugraz.sysds.lops.Unary.OperationTypes.SUBTRACT_NZ);
		HopsOpOp2LopsU.put(OpOp2.LOG_NZ, org.tugraz.sysds.lops.Unary.OperationTypes.LOG_NZ);
		HopsOpOp2LopsU.put(OpOp2.BITWAND, Unary.OperationTypes.BW_AND);
		HopsOpOp2LopsU.put(OpOp2.BITWOR, Unary.OperationTypes.BW_OR);
		HopsOpOp2LopsU.put(OpOp2.BITWXOR, Unary.OperationTypes.BW_XOR);
		HopsOpOp2LopsU.put(OpOp2.BITWSHIFTL, Unary.OperationTypes.BW_SHIFTL);
		HopsOpOp2LopsU.put(OpOp2.BITWSHIFTR, Unary.OperationTypes.BW_SHIFTR);
	}

	protected static final HashMap<Hop.OpOp1, org.tugraz.sysds.lops.Unary.OperationTypes> HopsOpOp1LopsU;
	static {
		HopsOpOp1LopsU = new HashMap<>();
		HopsOpOp1LopsU.put(OpOp1.NOT, org.tugraz.sysds.lops.Unary.OperationTypes.NOT);
		HopsOpOp1LopsU.put(OpOp1.ABS, org.tugraz.sysds.lops.Unary.OperationTypes.ABS);
		HopsOpOp1LopsU.put(OpOp1.SIN, org.tugraz.sysds.lops.Unary.OperationTypes.SIN);
		HopsOpOp1LopsU.put(OpOp1.COS, org.tugraz.sysds.lops.Unary.OperationTypes.COS);
		HopsOpOp1LopsU.put(OpOp1.TAN, org.tugraz.sysds.lops.Unary.OperationTypes.TAN);
		HopsOpOp1LopsU.put(OpOp1.ASIN, org.tugraz.sysds.lops.Unary.OperationTypes.ASIN);
		HopsOpOp1LopsU.put(OpOp1.ACOS, org.tugraz.sysds.lops.Unary.OperationTypes.ACOS);
		HopsOpOp1LopsU.put(OpOp1.ATAN, org.tugraz.sysds.lops.Unary.OperationTypes.ATAN);
		HopsOpOp1LopsU.put(OpOp1.SINH, org.tugraz.sysds.lops.Unary.OperationTypes.SINH);
		HopsOpOp1LopsU.put(OpOp1.COSH, org.tugraz.sysds.lops.Unary.OperationTypes.COSH);
		HopsOpOp1LopsU.put(OpOp1.TANH, org.tugraz.sysds.lops.Unary.OperationTypes.TANH);
		HopsOpOp1LopsU.put(OpOp1.SIGN, org.tugraz.sysds.lops.Unary.OperationTypes.SIGN);
		HopsOpOp1LopsU.put(OpOp1.SQRT, org.tugraz.sysds.lops.Unary.OperationTypes.SQRT);
		HopsOpOp1LopsU.put(OpOp1.EXP, org.tugraz.sysds.lops.Unary.OperationTypes.EXP);
		HopsOpOp1LopsU.put(OpOp1.LOG, org.tugraz.sysds.lops.Unary.OperationTypes.LOG);
		HopsOpOp1LopsU.put(OpOp1.ROUND, org.tugraz.sysds.lops.Unary.OperationTypes.ROUND);
		HopsOpOp1LopsU.put(OpOp1.CEIL, org.tugraz.sysds.lops.Unary.OperationTypes.CEIL);
		HopsOpOp1LopsU.put(OpOp1.FLOOR, org.tugraz.sysds.lops.Unary.OperationTypes.FLOOR);
		HopsOpOp1LopsU.put(OpOp1.CUMSUM, org.tugraz.sysds.lops.Unary.OperationTypes.CUMSUM);
		HopsOpOp1LopsU.put(OpOp1.CUMPROD, org.tugraz.sysds.lops.Unary.OperationTypes.CUMPROD);
		HopsOpOp1LopsU.put(OpOp1.CUMMIN, org.tugraz.sysds.lops.Unary.OperationTypes.CUMMIN);
		HopsOpOp1LopsU.put(OpOp1.CUMMAX, org.tugraz.sysds.lops.Unary.OperationTypes.CUMMAX);
		HopsOpOp1LopsU.put(OpOp1.CUMSUMPROD, org.tugraz.sysds.lops.Unary.OperationTypes.CUMSUMPROD);
		HopsOpOp1LopsU.put(OpOp1.INVERSE, org.tugraz.sysds.lops.Unary.OperationTypes.INVERSE);
		HopsOpOp1LopsU.put(OpOp1.CHOLESKY, org.tugraz.sysds.lops.Unary.OperationTypes.CHOLESKY);
		HopsOpOp1LopsU.put(OpOp1.CAST_AS_SCALAR, org.tugraz.sysds.lops.Unary.OperationTypes.NOTSUPPORTED);
		HopsOpOp1LopsU.put(OpOp1.CAST_AS_MATRIX, org.tugraz.sysds.lops.Unary.OperationTypes.NOTSUPPORTED);
		HopsOpOp1LopsU.put(OpOp1.SPROP, org.tugraz.sysds.lops.Unary.OperationTypes.SPROP);
		HopsOpOp1LopsU.put(OpOp1.SIGMOID, org.tugraz.sysds.lops.Unary.OperationTypes.SIGMOID);
		HopsOpOp1LopsU.put(OpOp1.LOG_NZ, org.tugraz.sysds.lops.Unary.OperationTypes.LOG_NZ);
		HopsOpOp1LopsU.put(OpOp1.CAST_AS_MATRIX, org.tugraz.sysds.lops.Unary.OperationTypes.CAST_AS_MATRIX);
		HopsOpOp1LopsU.put(OpOp1.CAST_AS_FRAME, org.tugraz.sysds.lops.Unary.OperationTypes.CAST_AS_FRAME);
	}

	protected static final HashMap<Hop.OpOp1, org.tugraz.sysds.lops.UnaryCP.OperationTypes> HopsOpOp1LopsUS;
	static {
		HopsOpOp1LopsUS = new HashMap<>();
		HopsOpOp1LopsUS.put(OpOp1.NOT, org.tugraz.sysds.lops.UnaryCP.OperationTypes.NOT);
		HopsOpOp1LopsUS.put(OpOp1.ABS, org.tugraz.sysds.lops.UnaryCP.OperationTypes.ABS);
		HopsOpOp1LopsUS.put(OpOp1.SIN, org.tugraz.sysds.lops.UnaryCP.OperationTypes.SIN);
		HopsOpOp1LopsUS.put(OpOp1.COS, org.tugraz.sysds.lops.UnaryCP.OperationTypes.COS);
		HopsOpOp1LopsUS.put(OpOp1.TAN, org.tugraz.sysds.lops.UnaryCP.OperationTypes.TAN);
		HopsOpOp1LopsUS.put(OpOp1.ASIN, org.tugraz.sysds.lops.UnaryCP.OperationTypes.ASIN);
		HopsOpOp1LopsUS.put(OpOp1.ACOS, org.tugraz.sysds.lops.UnaryCP.OperationTypes.ACOS);
		HopsOpOp1LopsUS.put(OpOp1.ATAN, org.tugraz.sysds.lops.UnaryCP.OperationTypes.ATAN);
		HopsOpOp1LopsUS.put(OpOp1.SINH, org.tugraz.sysds.lops.UnaryCP.OperationTypes.SINH);
		HopsOpOp1LopsUS.put(OpOp1.COSH, org.tugraz.sysds.lops.UnaryCP.OperationTypes.COSH);
		HopsOpOp1LopsUS.put(OpOp1.TANH, org.tugraz.sysds.lops.UnaryCP.OperationTypes.TANH);
		HopsOpOp1LopsUS.put(OpOp1.SQRT, org.tugraz.sysds.lops.UnaryCP.OperationTypes.SQRT);
		HopsOpOp1LopsUS.put(OpOp1.EXP, org.tugraz.sysds.lops.UnaryCP.OperationTypes.EXP);
		HopsOpOp1LopsUS.put(OpOp1.LOG, org.tugraz.sysds.lops.UnaryCP.OperationTypes.LOG);
		HopsOpOp1LopsUS.put(OpOp1.CAST_AS_SCALAR, org.tugraz.sysds.lops.UnaryCP.OperationTypes.CAST_AS_SCALAR);
		HopsOpOp1LopsUS.put(OpOp1.CAST_AS_MATRIX, org.tugraz.sysds.lops.UnaryCP.OperationTypes.CAST_AS_MATRIX);
		HopsOpOp1LopsUS.put(OpOp1.CAST_AS_FRAME, org.tugraz.sysds.lops.UnaryCP.OperationTypes.CAST_AS_FRAME);
		HopsOpOp1LopsUS.put(OpOp1.CAST_AS_DOUBLE, org.tugraz.sysds.lops.UnaryCP.OperationTypes.CAST_AS_DOUBLE);
		HopsOpOp1LopsUS.put(OpOp1.CAST_AS_INT, org.tugraz.sysds.lops.UnaryCP.OperationTypes.CAST_AS_INT);
		HopsOpOp1LopsUS.put(OpOp1.CAST_AS_BOOLEAN, org.tugraz.sysds.lops.UnaryCP.OperationTypes.CAST_AS_BOOLEAN);
		HopsOpOp1LopsUS.put(OpOp1.NROW, org.tugraz.sysds.lops.UnaryCP.OperationTypes.NROW);
		HopsOpOp1LopsUS.put(OpOp1.NCOL, org.tugraz.sysds.lops.UnaryCP.OperationTypes.NCOL);
		HopsOpOp1LopsUS.put(OpOp1.LENGTH, org.tugraz.sysds.lops.UnaryCP.OperationTypes.LENGTH);
		HopsOpOp1LopsUS.put(OpOp1.EXISTS, org.tugraz.sysds.lops.UnaryCP.OperationTypes.EXISTS);
		HopsOpOp1LopsUS.put(OpOp1.LINEAGE, org.tugraz.sysds.lops.UnaryCP.OperationTypes.LINEAGE);
		HopsOpOp1LopsUS.put(OpOp1.PRINT, org.tugraz.sysds.lops.UnaryCP.OperationTypes.PRINT);
		HopsOpOp1LopsUS.put(OpOp1.ASSERT, org.tugraz.sysds.lops.UnaryCP.OperationTypes.ASSERT);
		HopsOpOp1LopsUS.put(OpOp1.ROUND, org.tugraz.sysds.lops.UnaryCP.OperationTypes.ROUND);
		HopsOpOp1LopsUS.put(OpOp1.CEIL, org.tugraz.sysds.lops.UnaryCP.OperationTypes.CEIL);
		HopsOpOp1LopsUS.put(OpOp1.FLOOR, org.tugraz.sysds.lops.UnaryCP.OperationTypes.FLOOR);
		HopsOpOp1LopsUS.put(OpOp1.STOP, org.tugraz.sysds.lops.UnaryCP.OperationTypes.STOP);
	}

	protected static final HashMap<OpOp3, Ternary.OperationType> HopsOpOp3Lops;
	static {
		HopsOpOp3Lops = new HashMap<>();
		HopsOpOp3Lops.put(OpOp3.PLUS_MULT, Ternary.OperationType.PLUS_MULT);
		HopsOpOp3Lops.put(OpOp3.MINUS_MULT, Ternary.OperationType.MINUS_MULT);
		HopsOpOp3Lops.put(OpOp3.IFELSE, Ternary.OperationType.IFELSE);
	}
	
	/**
	 * Maps from a multiple (variable number of operands) Hop operation type to
	 * the corresponding Lop operation type. This is called in the MultipleOp
	 * constructLops() method that is used to construct the Lops that correspond
	 * to a Hop.
	 */
	protected static final HashMap<OpOpN, Nary.OperationType> HopsOpOpNLops;
	static {
		HopsOpOpNLops = new HashMap<>();
		HopsOpOpNLops.put(OpOpN.PRINTF, Nary.OperationType.PRINTF);
		HopsOpOpNLops.put(OpOpN.CBIND, Nary.OperationType.CBIND);
		HopsOpOpNLops.put(OpOpN.RBIND, Nary.OperationType.RBIND);
		HopsOpOpNLops.put(OpOpN.MIN, Nary.OperationType.MIN);
		HopsOpOpNLops.put(OpOpN.MAX, Nary.OperationType.MAX);
		HopsOpOpNLops.put(OpOpN.EVAL, Nary.OperationType.EVAL);
		HopsOpOpNLops.put(OpOpN.LIST, Nary.OperationType.LIST);
	}

	protected static final HashMap<OpOp1, String> HopsOpOp12String;
	protected static final HashMap<String, OpOp1> HopsStringOpOp1;
	
	static {
		HopsOpOp12String = new HashMap<>();
		HopsOpOp12String.put(OpOp1.ABS, "abs");
		HopsOpOp12String.put(OpOp1.CAST_AS_SCALAR, "castAsScalar");
		HopsOpOp12String.put(OpOp1.COS, "cos");
		HopsOpOp12String.put(OpOp1.EIGEN, "eigen");
		HopsOpOp12String.put(OpOp1.SVD, "svd");
		HopsOpOp12String.put(OpOp1.EXP, "exp");
		HopsOpOp12String.put(OpOp1.IQM, "iqm");
		HopsOpOp12String.put(OpOp1.MEDIAN, "median");
		HopsOpOp12String.put(OpOp1.LENGTH, "length");
		HopsOpOp12String.put(OpOp1.LOG, "log");
		HopsOpOp12String.put(OpOp1.NCOL, "ncol");
		HopsOpOp12String.put(OpOp1.NOT, "!");
		HopsOpOp12String.put(OpOp1.NROW, "nrow");
		HopsOpOp12String.put(OpOp1.PRINT, "print");
		HopsOpOp12String.put(OpOp1.ASSERT, "assert");
		HopsOpOp12String.put(OpOp1.ROUND, "round");
		HopsOpOp12String.put(OpOp1.SIN, "sin");
		HopsOpOp12String.put(OpOp1.SQRT, "sqrt");
		HopsOpOp12String.put(OpOp1.TAN, "tan");
		HopsOpOp12String.put(OpOp1.ASIN, "asin");
		HopsOpOp12String.put(OpOp1.ACOS, "acos");
		HopsOpOp12String.put(OpOp1.ATAN, "atan");
		HopsOpOp12String.put(OpOp1.SINH, "sinh");
		HopsOpOp12String.put(OpOp1.COSH, "cosh");
		HopsOpOp12String.put(OpOp1.TANH, "tanh");
		HopsOpOp12String.put(OpOp1.STOP, "stop");
		HopsOpOp12String.put(OpOp1.INVERSE, "inv");
		HopsOpOp12String.put(OpOp1.SPROP, "sprop");
		HopsOpOp12String.put(OpOp1.SIGMOID, "sigmoid");
		
		HopsStringOpOp1 = new HashMap<>();
		for( Entry<OpOp1,String> e : HopsOpOp12String.entrySet() )
			HopsStringOpOp1.put(e.getValue(), e.getKey());
	}

	public static OpOp1 getUnaryOpCode(String op) {
		return HopsStringOpOp1.get(op);
	}
	
	protected static final HashMap<Hop.ParamBuiltinOp, org.tugraz.sysds.lops.ParameterizedBuiltin.OperationTypes> HopsParameterizedBuiltinLops;
	static {
		HopsParameterizedBuiltinLops = new HashMap<>();
		HopsParameterizedBuiltinLops.put(ParamBuiltinOp.CDF, ParameterizedBuiltin.OperationTypes.CDF);
		HopsParameterizedBuiltinLops.put(ParamBuiltinOp.INVCDF, ParameterizedBuiltin.OperationTypes.INVCDF);
		HopsParameterizedBuiltinLops.put(ParamBuiltinOp.RMEMPTY, ParameterizedBuiltin.OperationTypes.RMEMPTY);
		HopsParameterizedBuiltinLops.put(ParamBuiltinOp.REPLACE, ParameterizedBuiltin.OperationTypes.REPLACE);
		HopsParameterizedBuiltinLops.put(ParamBuiltinOp.REXPAND, ParameterizedBuiltin.OperationTypes.REXPAND);
		HopsParameterizedBuiltinLops.put(ParamBuiltinOp.LOWER_TRI, ParameterizedBuiltin.OperationTypes.LOWER_TRI);
		HopsParameterizedBuiltinLops.put(ParamBuiltinOp.UPPER_TRI, ParameterizedBuiltin.OperationTypes.UPPER_TRI);
		HopsParameterizedBuiltinLops.put(ParamBuiltinOp.TRANSFORMAPPLY, ParameterizedBuiltin.OperationTypes.TRANSFORMAPPLY);
		HopsParameterizedBuiltinLops.put(ParamBuiltinOp.TRANSFORMDECODE, ParameterizedBuiltin.OperationTypes.TRANSFORMDECODE);
		HopsParameterizedBuiltinLops.put(ParamBuiltinOp.TRANSFORMCOLMAP, ParameterizedBuiltin.OperationTypes.TRANSFORMCOLMAP);
		HopsParameterizedBuiltinLops.put(ParamBuiltinOp.TRANSFORMMETA, ParameterizedBuiltin.OperationTypes.TRANSFORMMETA);
		HopsParameterizedBuiltinLops.put(ParamBuiltinOp.TOSTRING, ParameterizedBuiltin.OperationTypes.TOSTRING);
		HopsParameterizedBuiltinLops.put(ParamBuiltinOp.LIST, ParameterizedBuiltin.OperationTypes.LIST);
		HopsParameterizedBuiltinLops.put(ParamBuiltinOp.PARAMSERV, ParameterizedBuiltin.OperationTypes.PARAMSERV);
	}

	protected static final HashMap<OpOp2, String> HopsOpOp2String;
	protected static final HashMap<String,OpOp2> HopsStringOpOp2;
	static {
		HopsOpOp2String = new HashMap<>();
		HopsOpOp2String.put(OpOp2.PLUS, "+");
		HopsOpOp2String.put(OpOp2.MINUS, "-");
		HopsOpOp2String.put(OpOp2.MINUS_NZ, "-nz");
		HopsOpOp2String.put(OpOp2.MINUS1_MULT, "-1*");
		HopsOpOp2String.put(OpOp2.MULT, "*");
		HopsOpOp2String.put(OpOp2.DIV, "/");
		HopsOpOp2String.put(OpOp2.MODULUS, "%%");
		HopsOpOp2String.put(OpOp2.INTDIV, "%/%");
		HopsOpOp2String.put(OpOp2.MIN, "min");
		HopsOpOp2String.put(OpOp2.MAX, "max");
		HopsOpOp2String.put(OpOp2.LESSEQUAL, "<=");
		HopsOpOp2String.put(OpOp2.LESS, "<");
		HopsOpOp2String.put(OpOp2.GREATEREQUAL, ">=");
		HopsOpOp2String.put(OpOp2.GREATER, ">");
		HopsOpOp2String.put(OpOp2.EQUAL, "==");
		HopsOpOp2String.put(OpOp2.NOTEQUAL, "!=");
		HopsOpOp2String.put(OpOp2.OR, "|");
		HopsOpOp2String.put(OpOp2.AND, "&");
		HopsOpOp2String.put(OpOp2.LOG, "log");
		HopsOpOp2String.put(OpOp2.LOG_NZ, "log_nz");
		HopsOpOp2String.put(OpOp2.POW, "^");
		HopsOpOp2String.put(OpOp2.CONCAT, "concat");
		HopsOpOp2String.put(OpOp2.INVALID, "?");
		HopsOpOp2String.put(OpOp2.QUANTILE, "quantile");
		HopsOpOp2String.put(OpOp2.INTERQUANTILE, "interquantile");
		HopsOpOp2String.put(OpOp2.IQM, "IQM");
		HopsOpOp2String.put(OpOp2.MEDIAN, "median");
		HopsOpOp2String.put(OpOp2.MOMENT, "cm");
		HopsOpOp2String.put(OpOp2.COV, "cov");
		HopsOpOp2String.put(OpOp2.CBIND, "cbind");
		HopsOpOp2String.put(OpOp2.RBIND, "rbind");
		HopsOpOp2String.put(OpOp2.SOLVE, "solve");
		HopsOpOp2String.put(OpOp2.XOR, "xor");
		HopsOpOp2String.put(OpOp2.BITWAND, "bitwAnd");
		HopsOpOp2String.put(OpOp2.BITWOR,  "bitwOr");
		HopsOpOp2String.put(OpOp2.BITWXOR, "bitwXor");
		HopsOpOp2String.put(OpOp2.BITWSHIFTL, "bitwShiftL");
		HopsOpOp2String.put(OpOp2.BITWSHIFTR, "bitwShiftR");
		
		HopsStringOpOp2 = new HashMap<>();
		for( Entry<OpOp2,String> e : HopsOpOp2String.entrySet() )
			HopsStringOpOp2.put(e.getValue(), e.getKey());
	}
	
	public static String getBinaryOpCode(OpOp2 op) {
		return HopsOpOp2String.get(op);
	}
	
	public static OpOp2 getBinaryOpCode(String op) {
		return HopsStringOpOp2.get(op);
	}
	
	protected static final HashMap<Hop.OpOp3, String> HopsOpOp3String;
	protected static final HashMap<String,OpOp3> HopsStringOpOp3;
	static {
		HopsOpOp3String = new HashMap<>();
		HopsOpOp3String.put(OpOp3.QUANTILE, "quantile");
		HopsOpOp3String.put(OpOp3.INTERQUANTILE, "interquantile");
		HopsOpOp3String.put(OpOp3.CTABLE, "ctable");
		HopsOpOp3String.put(OpOp3.MOMENT, "cm");
		HopsOpOp3String.put(OpOp3.COV, "cov");
		HopsOpOp3String.put(OpOp3.PLUS_MULT, "+*");
		HopsOpOp3String.put(OpOp3.MINUS_MULT, "-*");
		HopsOpOp3String.put(OpOp3.IFELSE, "ifelse");
		
		HopsStringOpOp3 = new HashMap<>();
		for( Entry<OpOp3,String> e : HopsOpOp3String.entrySet() )
			HopsStringOpOp3.put(e.getValue(), e.getKey());
	}
	
	public static String getTernaryOpCode(OpOp3 op) {
		return HopsOpOp3String.get(op);
	}
	
	public static OpOp3 getTernaryOpCode(String op) {
		return HopsStringOpOp3.get(op);
	}
	
	protected static final HashMap<Hop.OpOp4, String> HopsOpOp4String;
	static {
		HopsOpOp4String = new HashMap<>();
		HopsOpOp4String.put(OpOp4.WSLOSS,   "wsloss");
		HopsOpOp4String.put(OpOp4.WSIGMOID, "wsigmoid");
		HopsOpOp4String.put(OpOp4.WCEMM,    "wcemm");
		HopsOpOp4String.put(OpOp4.WDIVMM,   "wdivmm");
		HopsOpOp4String.put(OpOp4.WUMM,     "wumm");
	}

	protected static final HashMap<Hop.Direction, String> HopsDirection2String;
	static {
		HopsDirection2String = new HashMap<>();
		HopsDirection2String.put(Direction.RowCol, "RC");
		HopsDirection2String.put(Direction.Col, "C");
		HopsDirection2String.put(Direction.Row, "R");
	}

	protected static final HashMap<Hop.AggOp, String> HopsAgg2String;
	static {
		HopsAgg2String = new HashMap<>();
		HopsAgg2String.put(AggOp.SUM, "+");
		HopsAgg2String.put(AggOp.SUM_SQ, "sq+");
		HopsAgg2String.put(AggOp.PROD, "*");
		HopsAgg2String.put(AggOp.MIN, "min");
		HopsAgg2String.put(AggOp.MAX, "max");
		HopsAgg2String.put(AggOp.MAXINDEX, "maxindex");
		HopsAgg2String.put(AggOp.MININDEX, "minindex");
		HopsAgg2String.put(AggOp.TRACE, "trace");
		HopsAgg2String.put(AggOp.MEAN, "mean");
		HopsAgg2String.put(AggOp.VAR, "var");
	}

	protected static final HashMap<Hop.ReOrgOp, String> HopsTransf2String;
	static {
		HopsTransf2String = new HashMap<>();
		HopsTransf2String.put(ReOrgOp.TRANS, "t");
		HopsTransf2String.put(ReOrgOp.DIAG, "diag");
		HopsTransf2String.put(ReOrgOp.RESHAPE, "rshape");
		HopsTransf2String.put(ReOrgOp.SORT, "sort");
	}

	protected static final HashMap<DataOpTypes, String> HopsData2String;
	static {
		HopsData2String = new HashMap<>();
		HopsData2String.put(DataOpTypes.PERSISTENTREAD, "PRead");
		HopsData2String.put(DataOpTypes.PERSISTENTWRITE, "PWrite");
		HopsData2String.put(DataOpTypes.TRANSIENTWRITE, "TWrite");
		HopsData2String.put(DataOpTypes.TRANSIENTREAD, "TRead");
		HopsData2String.put(DataOpTypes.FUNCTIONOUTPUT, "FunOut");
	}

	public static OpOp2 getOpOp2ForOuterVectorOperation(String op) 
	{
		if( "+".equals(op) ) return OpOp2.PLUS;
		else if( "-".equals(op) ) return OpOp2.MINUS;
		else if( "*".equals(op) ) return OpOp2.MULT;
		else if( "/".equals(op) ) return OpOp2.DIV;
		else if( "%%".equals(op) ) return OpOp2.MODULUS;
		else if( "%/%".equals(op) ) return OpOp2.INTDIV;
		else if( "min".equals(op) ) return OpOp2.MIN;
		else if( "max".equals(op) ) return OpOp2.MAX;
		else if( "<=".equals(op) ) return OpOp2.LESSEQUAL;
		else if( "<".equals(op) ) return OpOp2.LESS;
		else if( ">=".equals(op) ) return OpOp2.GREATEREQUAL;
		else if( ">".equals(op) ) return OpOp2.GREATER;
		else if( "==".equals(op) ) return OpOp2.EQUAL;
		else if( "!=".equals(op) ) return OpOp2.NOTEQUAL;
		else if( "|".equals(op) ) return OpOp2.OR;
		else if( "xor".equals(op) ) return OpOp2.XOR;
		else if( "&".equals(op) ) return OpOp2.AND;
		else if( "log".equals(op) ) return OpOp2.LOG;
		else if( "^".equals(op) ) return OpOp2.POW;
		else if("bitwAnd".equals(op) ) return OpOp2.BITWAND;
		else if("bitwOr".equals(op) ) return OpOp2.BITWOR;
		else if("bitwXor".equals(op) ) return OpOp2.BITWXOR;
		else if("bitwShiftL".equals(op) ) return OpOp2.BITWSHIFTL;
		else if("bitwShiftR".equals(op) ) return OpOp2.BITWSHIFTR;
		
		return null;
	}

	/////////////////////////////////////
	// methods for dynamic re-compilation
	/////////////////////////////////////

	/**
	 * Indicates if dynamic recompilation is required for this hop. 
	 * 
	 * @return true if dynamic recompilation required
	 */
	public boolean requiresRecompile() {
		return _requiresRecompile;
	}
	
	/**
	 * Marks the hop for dynamic recompilation. 
	 */
	public void setRequiresRecompile() {
		_requiresRecompile = true;
	}
	
	/**
	 * Marks the hop for dynamic recompilation, if dynamic recompilation is 
	 * enabled and one of the three basic scenarios apply:
	 * <ul>
	 *  <li> The hop has unknown dimensions or sparsity and is scheduled for 
	 *    remote execution, in which case the latency for distributed jobs easily 
	 *    covers any recompilation overheads. </li>
	 *  <li> The hop has unknown dimensions and is scheduled for local execution 
	 *    due to forced single node execution type. </li>
	 *  <li> The hop has unknown dimensions and is scheduled for local execution 
	 *    due to good worst-case memory estimates but codegen is enabled, which
	 *    requires (mostly) known sizes to validity conditions and cost estimation. </li>
	 * <ul> <p>
	 */
	protected void setRequiresRecompileIfNecessary() {
		boolean caseRemote = (!dimsKnown(true) && _etype == ExecType.SPARK);
		boolean caseLocal = (!dimsKnown() && _etypeForced == ExecType.CP);
		boolean caseCodegen = (!dimsKnown() && ConfigurationManager.isCodegenEnabled());
		
		if( ConfigurationManager.isDynamicRecompilation() 
			&& (caseRemote || caseLocal || caseCodegen) )
			setRequiresRecompile();
	}

	/**
	 * Update the output size information for this hop.
	 */
	public abstract void refreshSizeInformation();
	
	/**
	 * Util function for refreshing scalar rows input parameter.
	 * 
	 * @param input high-level operator
	 */
	protected void refreshRowsParameterInformation( Hop input )
	{
		long size = computeSizeInformation(input);
			
		//always set the computed size not just if known (positive) in order to allow 
		//recompile with unknowns to reset sizes (otherwise potential for incorrect results)
		setDim1( size );
	}
	
	
	/**
	 * Util function for refreshing scalar cols input parameter.
	 * 
	 * @param input high-level operator
	 */
	protected void refreshColsParameterInformation( Hop input )
	{
		long size = computeSizeInformation(input);
		
		//always set the computed size not just if known (positive) in order to allow 
		//recompile with unknowns to reset sizes (otherwise potential for incorrect results)
		setDim2( size );
	}

	public static long computeSizeInformation( Hop input )
	{
		long ret = -1;
		
		try 
		{
			long tmp = OptimizerUtils.rEvalSimpleLongExpression(input, new HashMap<Long,Long>());
			if( tmp!=Long.MAX_VALUE )
				ret = tmp;
		}
		catch(Exception ex)
		{
			LOG.error("Failed to compute size information.", ex);
			ret = -1;
		}
		
		return ret;
	}
	
	//always set the computed size not just if known (positive) in order to allow 
	//recompile with unknowns to reset sizes (otherwise potential for incorrect results)
	
	public void refreshRowsParameterInformation( Hop input, LocalVariableMap vars ) {
		setDim1(computeSizeInformation(input, vars));
	}

	public void refreshRowsParameterInformation( Hop input, LocalVariableMap vars, HashMap<Long,Long> memo ) {
		setDim1(computeSizeInformation(input, vars, memo));
	}
	
	public void refreshColsParameterInformation( Hop input, LocalVariableMap vars ) {
		setDim2(computeSizeInformation(input, vars));
	}
	
	public void refreshColsParameterInformation( Hop input, LocalVariableMap vars, HashMap<Long,Long> memo ) {
		setDim2(computeSizeInformation(input, vars, memo));
	}

	public long computeSizeInformation( Hop input, LocalVariableMap vars ) {
		return computeSizeInformation(input, vars, new HashMap<Long,Long>());
	}
	
	public long computeSizeInformation( Hop input, LocalVariableMap vars, HashMap<Long,Long> memo )
	{
		long ret = -1;
		try {
			long tmp = OptimizerUtils.rEvalSimpleLongExpression(input, memo, vars);
			if( tmp!=Long.MAX_VALUE )
				ret = tmp;
		}
		catch(Exception ex) {
			LOG.error("Failed to compute size information.", ex);
			ret = -1;
		}
		return ret;
	}

	public double computeBoundsInformation( Hop input ) {
		double ret = Double.MAX_VALUE;
		try {
			ret = OptimizerUtils.rEvalSimpleDoubleExpression(input, new HashMap<Long, Double>());
		}
		catch(Exception ex) {
			LOG.error("Failed to compute bounds information.", ex);
			ret = Double.MAX_VALUE;
		}
		return ret;
	}
	
	public final double computeBoundsInformation( Hop input, LocalVariableMap vars ) {
		return computeBoundsInformation(input, vars, new HashMap<Long, Double>());
	}
	
	public final double computeBoundsInformation( Hop input, LocalVariableMap vars, HashMap<Long, Double> memo ) {
		double ret = Double.MAX_VALUE;
		try {
			ret = OptimizerUtils.rEvalSimpleDoubleExpression(input, memo, vars);
		}
		catch(Exception ex) {
			LOG.error("Failed to compute bounds information.", ex);
			ret = Double.MAX_VALUE;
		}
		return ret;
	}
	
	/**
	 * Compute worst case estimate for size expression based on worst-case
	 * statistics of inputs. Limited set of supported operations in comparison
	 * to refresh rows/cols.
	 * 
	 * @param input high-level operator
	 * @param memo memory table
	 * @return worst case estimate for size expression
	 */
	protected long computeDimParameterInformation( Hop input, MemoTable memo )
	{
		long ret = -1;
		
		if( input instanceof UnaryOp )
		{
			if( ((UnaryOp)input).getOp() == Hop.OpOp1.NROW ) {
				DataCharacteristics mc = memo.getAllInputStats(input.getInput().get(0));
				if( mc.rowsKnown() )
					ret = mc.getRows();
			}
			else if ( ((UnaryOp)input).getOp() == Hop.OpOp1.NCOL ) {
				DataCharacteristics mc = memo.getAllInputStats(input.getInput().get(0));
				if( mc.colsKnown() )
					ret = mc.getCols();
			}
		}
		else if ( input instanceof LiteralOp )
		{
			ret = UtilFunctions.parseToLong(input.getName());
		}
		else if ( input instanceof BinaryOp )
		{
			long dim = rEvalSimpleBinaryLongExpression(input, new HashMap<Long, Long>(), memo);
			if( dim != Long.MAX_VALUE ) //if known
				ret = dim ;
		}
		
		return ret;
	}

	protected long rEvalSimpleBinaryLongExpression( Hop root, HashMap<Long, Long> valMemo, MemoTable memo )
	{
		//memoization (prevent redundant computation of common subexpr)
		if( valMemo.containsKey(root.getHopID()) )
			return valMemo.get(root.getHopID());
		
		long ret = Long.MAX_VALUE;
		
		if( root instanceof LiteralOp )
		{
			long dim = UtilFunctions.parseToLong(root.getName());
			if( dim != -1 ) //if known
				ret = dim;
		}
		else if( root instanceof UnaryOp )
		{
			UnaryOp uroot = (UnaryOp) root;
			long dim = -1;
			if(uroot.getOp() == Hop.OpOp1.NROW)
			{
				DataCharacteristics mc = memo.getAllInputStats(uroot.getInput().get(0));
				dim = mc.getRows();
			}
			else if( uroot.getOp() == Hop.OpOp1.NCOL )
			{
				DataCharacteristics mc = memo.getAllInputStats(uroot.getInput().get(0));
				dim = mc.getCols();
			}
			if( dim != -1 ) //if known
				ret = dim;
		}
		else if( root instanceof BinaryOp )
		{ 
			if( OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION )
			{
				BinaryOp broot = (BinaryOp) root;
				long lret = rEvalSimpleBinaryLongExpression(broot.getInput().get(0), valMemo, memo);
				long rret = rEvalSimpleBinaryLongExpression(broot.getInput().get(1), valMemo, memo);
				//note: positive and negative values might be valid subexpressions
				if( lret!=Long.MAX_VALUE && rret!=Long.MAX_VALUE ) //if known
				{
					switch( broot.getOp() )
					{
						case PLUS:	ret = lret + rret; break;
						case MULT:  ret = lret * rret; break;
						case MIN:   ret = Math.min(lret, rret); break;
						case MAX:   ret = Math.max(lret, rret); break;
						default:    ret = Long.MAX_VALUE;
					}
				}
				//exploit min constraint to propagate 
				else if( broot.getOp()==OpOp2.MIN && (lret!=Double.MAX_VALUE || rret!=Double.MAX_VALUE) )
				{
					ret = Math.min(lret, rret);
				}
			}
		}
		
		valMemo.put(root.getHopID(), ret);
		return ret;
	}

	/**
	 * Clones the attributes of that and copies it over to this.
	 * 
	 * @param that high-level operator
	 * @param withRefs true if with references
	 * @throws CloneNotSupportedException if CloneNotSupportedException occurs
	 */
	protected void clone( Hop that, boolean withRefs ) 
		throws CloneNotSupportedException 
	{
		if( withRefs )
			throw new CloneNotSupportedException( "Hops deep copy w/ lops/inputs/parents not supported." );
		
		_name = that._name;
		_dataType = that._dataType;
		_valueType = that._valueType;
		_visited = that._visited;
		_dim1 = that._dim1;
		_dim2 = that._dim2;
		_blocksize = that._blocksize;
		_nnz = that._nnz;
		_updateType = that._updateType;

		//no copy of lops (regenerated)
		_parent = new ArrayList<>(_parent.size());
		_input = new ArrayList<>(_input.size());
		_lops = null;
		
		_etype = that._etype;
		_etypeForced = that._etypeForced;
		_outputMemEstimate = that._outputMemEstimate;
		_memEstimate = that._memEstimate;
		_processingMemEstimate = that._processingMemEstimate;
		_requiresRecompile = that._requiresRecompile;
		_requiresReblock = that._requiresReblock;
		_requiresCheckpoint = that._requiresCheckpoint;
		_outputEmptyBlocks = that._outputEmptyBlocks;
		
		_beginLine = that._beginLine;
		_beginColumn = that._beginColumn;
		_endLine = that._endLine;
		_endColumn = that._endColumn;
	}
	
	@Override
	public abstract Object clone() throws CloneNotSupportedException;
	
	public abstract boolean compare( Hop that );
	
	
	
	///////////////////////////////////////////////////////////////////////////
	// store position information for Hops
	///////////////////////////////////////////////////////////////////////////
	public int _beginLine, _beginColumn;
	public int _endLine, _endColumn;
	public String _filename;
	public String _text;
	
	public void setBeginLine(int passed)    { _beginLine = passed;   }
	public void setBeginColumn(int passed) 	{ _beginColumn = passed; }
	public void setEndLine(int passed) 		{ _endLine = passed;   }
	public void setEndColumn(int passed)	{ _endColumn = passed; }
	public void setFilename(String passed) { _filename = passed; }
	public void setText(String text) { _text = text; }

	public int getBeginLine()	{ return _beginLine;   }
	public int getBeginColumn() { return _beginColumn; }
	public int getEndLine() 	{ return _endLine;   }
	public int getEndColumn()	{ return _endColumn; }
	public String getFilename()	{ return _filename; }
	public String getText() { return _text; }
	
	public String printErrorLocation(){
		if(_filename != null)
			return "ERROR: " + _filename + " line " + _beginLine + ", column " + _beginColumn + " -- ";
		else
			return "ERROR: line " + _beginLine + ", column " + _beginColumn + " -- ";
	}

	/**
	 * Sets the linenumbers of this hop to a given lop.
	 * 
	 * @param lop low-level operator
	 */
	protected void setLineNumbers(Lop lop)
	{
		lop.setAllPositions(this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	}

	/**
	 * Set parse information.
	 *
	 * @param parseInfo
	 *            parse information, such as beginning line position, beginning
	 *            column position, ending line position, ending column position,
	 *            text, and filename
	 */
	public void setParseInfo(ParseInfo parseInfo) {
		_beginLine = parseInfo.getBeginLine();
		_beginColumn = parseInfo.getBeginColumn();
		_endLine = parseInfo.getEndLine();
		_endColumn = parseInfo.getEndColumn();
		_text = parseInfo.getText();
		_filename = parseInfo.getFilename();
	}

} // end class
