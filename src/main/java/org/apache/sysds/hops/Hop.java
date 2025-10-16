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

package org.apache.sysds.hops;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.cost.ComputeCost;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.hops.recompile.Recompiler.ResetType;
import org.apache.sysds.lops.CSVReBlock;
import org.apache.sysds.lops.Checkpoint;
import org.apache.sysds.lops.Compression;
import org.apache.sysds.lops.Data;
import org.apache.sysds.lops.DeCompression;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.LopsException;
import org.apache.sysds.lops.ReBlock;
import org.apache.sysds.lops.UnaryCP;
import org.apache.sysds.parser.ParseInfo;
import org.apache.sysds.runtime.compress.SingletonLookupHashMap;
import org.apache.sysds.runtime.compress.workload.AWTreeNode;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;
import org.apache.sysds.runtime.instructions.gpu.context.GPUContextPool;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * Hop is a High level operator, that is the first intermediate representation compiled from the definitions supplied in
 * DML.
 */
public abstract class Hop implements ParseInfo {
	protected static final Log LOG =  LogFactory.getLog(Hop.class.getName());

	public static final long CPThreshold = 2000;

	// static variable to assign an unique ID to every hop that is created
	private static IDSequence _seqHopID = new IDSequence();
	
	protected final long _ID;
	protected String _name;
	protected DataType _dataType;
	protected ValueType _valueType;
	protected boolean _visited = false;
	protected DataCharacteristics _dc = new MatrixCharacteristics();
	protected UpdateType _updateType = UpdateType.COPY;

	/** The output Hops that are connected to this Hop */
	protected List<Hop> _parent = new ArrayList<>();
	/** The input Hops that are connected to this Hop */
	protected List<Hop> _input = new ArrayList<>();

	/** Currently used exec type */
	protected ExecType _etype = null; 
	/** Exec type forced via platform or external optimizer */
	protected ExecType _etypeForced = null; 

	/**
	 * Field defining if the output of the operation should be federated.
	 * If it is fout, the output should be kept at federated sites.
	 * If it is lout, the output should be retrieved by the coordinator.
	 */
	protected FederatedOutput _federatedOutput = FederatedOutput.NONE;

	/**
	 * Field defining if prefetch should be activated for operation.
	 * When prefetch is activated, the output will be transferred from
	 * remote federated sites to local before one of the subsequent
	 * local operations.
	 */
	protected boolean activatePrefetch;
	
	/** Estimated size for the output produced from this Hop in bytes */
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

	/**
	 *  indicates if the output of this hop needs to be compressed
	 * (this happens on persistent reads after reblock but before checkpoint)
	*/
	protected boolean _requiresCompression = false;

	/** Boolean specifying if the output of this hop is compressed */
	protected boolean _compressedOutput = false;

	/** Compressed Size of this hop */
	protected long _compressedSize = 0;

	/** A WTree for this hop instruction in case the compression */
	protected AWTreeNode _compressedWorkloadTree = null;

	/** Boolean specifying if decompression is required.*/
	protected boolean _requiresDeCompression = false;
	
	// indicates if the output of this hop needs to be checkpointed (cached)
	// (the default storage level for caching is not yet exposed here)
	protected boolean _requiresCheckpoint = false;
	
	// indicates if the output of this hops needs to contain materialized empty blocks 
	// if those exists; otherwise only blocks w/ non-zero values are materialized
	protected boolean _outputEmptyBlocks = true;

	// indicates if the output of this hop needs to be saved in lineage cache
	// this is a suggestion by compiler and can be ignored by runtime
	protected boolean _requiresLineageCaching = true;
	
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

	public ExecType getExecType() {
		return _etype;
	}

	public void setExecType(ExecType execType){
		_etype = execType;
	}

	public void setFederatedOutput(FederatedOutput federatedOutput){
		_federatedOutput = federatedOutput;
	}

	/**
	 * Activate prefetch of HOP.
	 */
	public void activatePrefetch(){
		activatePrefetch = true;
	}

	public void deactivatePrefetch(){
		activatePrefetch = false;
	}

	/**
	 * Checks if prefetch is activated for this hop.
	 * @return true if prefetch is activated
	 */
	public boolean prefetchActivated(){
		return activatePrefetch;
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
		logForcedETCall(etype);
		_etypeForced = etype;
	}

	private void logForcedETCall(ExecType newEType){
		if ( LOG.isDebugEnabled() && _etypeForced != null && newEType != _etypeForced )
			LOG.debug("Forced ExecType of " + this + " changed from " + _etypeForced + " to " + newEType);
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
		else if (DMLScript.USE_OOC)
			_etypeForced = ExecType.OOC;
		else if ( DMLScript.getGlobalExecMode() == ExecMode.SINGLE_NODE && _etypeForced != ExecType.FED ) {
			if(OptimizerUtils.isMemoryBasedOptLevel() && DMLScript.USE_ACCELERATOR && isGPUEnabled()) {
				// enabled with -exec singlenode -gpu option
				_etypeForced = findExecTypeByMemEstimate();
				if(_etypeForced != ExecType.CP && _etypeForced != ExecType.GPU)
					_etypeForced = ExecType.CP;
			}
			else if (DMLScript.USE_OOC && !(this instanceof BinaryOp)){
				_etypeForced = ExecType.OOC;
			}
			else {
				// enabled with -exec singlenode option
				_etypeForced = ExecType.CP;
			}
		}
		else if ( DMLScript.getGlobalExecMode() == ExecMode.SPARK )
			_etypeForced = ExecType.SPARK; // enabled with -exec spark option
		else if ( DMLScript.getGlobalExecMode() == ExecMode.HYBRID
				&& ConfigurationManager.getCompilerConfigFlag(ConfigType.RESOURCE_OPTIMIZATION))
			_etypeForced = null;
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
		boolean invalid = !OptimizerUtils.isValidCPDimensions(_dc.getRows(), _dc.getCols());
		for( Hop in : getInput() )
			invalid |= !OptimizerUtils.isValidCPDimensions(in._dc.getRows(), in._dc.getCols());
		return !invalid;
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

	public void setRequiresCompression(){
		_requiresCompression = true;
	}

	public void setRequiresCompression(AWTreeNode node) {
		_requiresCompression = true;
		_compressedWorkloadTree = node;
	}

	public void setRequiresDeCompression(){
		_requiresDeCompression = true;
	}

	public boolean isRequiredDecompression(){
		return _requiresDeCompression;
	}
	
	public boolean requiresCompression() {
		return _requiresCompression;
	}

	public void setCompressedOutput(boolean value){
		_compressedOutput = value;
	}

	public void setCompressedSize(long size){
		_compressedSize = size;
	}

	public long getCompressedSize(){
		return _compressedSize;
	}

	public boolean isCompressedOutput(){
		return _compressedOutput;
	}

	public boolean hasCompressedInput(){
		for(Hop h : getInput()){
			if(h.isCompressedOutput())
				return true;
		}
		return false;
	}

	public long compressedSize(){
		return _compressedSize;
	}
	
	public void setRequiresLineageCaching(boolean flag) {
		_requiresLineageCaching = flag;
	}
	
	public boolean requiresLineageCaching() {
		return _requiresLineageCaching;
	}

	public void updateLopFedOut(Lop lop){
		updateLopFedOut(lop, getExecType(), _federatedOutput);
	}

	public void updateLopFedOut(Lop lop, ExecType execType, FederatedOutput fedOut){
		if ( execType == ExecType.FED )
			lop.setFederatedOutput(fedOut);
	}
	
	public void constructAndSetLopsDataFlowProperties() {
		//propagate federated output configuration to lops
		if( isFederated() || getLops().getExecType() == ExecType.FED )
			getLops().setFederatedOutput(_federatedOutput);
		if ( prefetchActivated() )
			getLops().activatePrefetch();

		//propagate compute and memory estimates to lops
		//FIXME: Compute cost. Handle multiple Lops from one Hop case
		if (ConfigurationManager.isAutoLinearizationEnabled())
			setMemoryAndComputeEstimates(getLops());
		
		//Step 1: construct reblock lop if required (output of hop)
		constructAndSetReblockLopIfRequired();
		
		//Step 2: construct compression lop if required
		constructAndSetCompressionLopIfRequired();

		//Step 3: construct checkpoint lop if required (output of hop or reblock)
		constructAndSetCheckpointLopIfRequired();
	}

	private void constructAndSetReblockLopIfRequired() 
	{
		//determine execution type
		ExecType et = DMLScript.USE_OOC ? ExecType.OOC : ExecType.CP;
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
			if(this instanceof DataOp // CSV
				&& ((DataOp) this).getOp() == OpOpData.PERSISTENTREAD &&
				((DataOp) this).getFileFormat() == FileFormat.CSV) {
				reblock = new CSVReBlock(input, getBlocksize(), getDataType(), getValueType(), et);
			}
			else { // ALL OTHER
				reblock = new ReBlock(input, getBlocksize(), getDataType(), getValueType(), _outputEmptyBlocks, et);
			}

			// replace this lop with the reblock instruction
			setOutputDimensions(reblock);
			setLineNumbers(reblock);
			setLops(reblock);
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
					double matrixPSize = OptimizerUtils.estimatePartitionedSizeExactSparsity(_dc);
					double dataCache = SparkExecutionContext.getDataMemoryBudget(true, true);
					serializedStorage = MatrixBlock.evalSparseFormatInMemory(_dc)
						&& matrixPSize > dataCache //sparse in-memory does not fit in agg mem 
						&& (OptimizerUtils.getSparsity(_dc) < MatrixBlock.ULTRA_SPARSITY_TURN_POINT
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

	protected void constructAndSetCompressionLopIfRequired() {
		if((requiresCompression()) ^ _requiresDeCompression){ // xor
			ExecType et = getExecutionModeForCompression();

			Lop compressionInstruction = null;
			
			//TODO generalize threads
			final int k = OptimizerUtils.getConstrainedNumThreads(-1); 
			if(requiresCompression()) {
				if(_compressedWorkloadTree != null) {
					SingletonLookupHashMap m = SingletonLookupHashMap.getMap();
					int singletonID = m.put(_compressedWorkloadTree);
					compressionInstruction = new Compression(getLops(), getDataType(), getValueType(), et, singletonID, k);
				}
				else
					compressionInstruction = new Compression(getLops(), getDataType(), getValueType(), et, 0, k);
			}
			else if(_requiresDeCompression && et != ExecType.SPARK) // Disabled spark decompression instruction.
				compressionInstruction = new DeCompression(getLops(), getDataType(), getValueType(), et);
			else
				return;

			setOutputDimensions( compressionInstruction );
			setLineNumbers( compressionInstruction );
			setLops( compressionInstruction );
		}
	}

	protected ExecType getExecutionModeForCompression(){
		ExecType et = ExecType.CP;
		// conditional checkpoint based on memory estimate in order to avoid unnecessary 
		// persist and unpersist calls (4x the memory budget is conservative)
		if( OptimizerUtils.isSparkExecutionMode() && getDataType()!=DataType.SCALAR )
			if( OptimizerUtils.isHybridExecutionMode() 
				&& 2 * _outputMemEstimate < OptimizerUtils.getLocalMemBudget()
				|| _etypeForced == ExecType.CP || getLops().isExecCP() )
				et = ExecType.CP;
			else 
				et = ExecType.SPARK;
		return et;
	}

	public static Lop createOffsetLop( Hop hop, boolean repCols ) {
		Lop offset = null;
		if( ConfigurationManager.isDynamicRecompilation() && hop.dimsKnown() ) {
			// If dynamic recompilation is enabled and dims are known, we can replace the ncol with 
			// a literal in order to increase the piggybacking potential. This is safe because append 
			// is always marked for recompilation and hence, we have propagated the exact dimensions.
			offset = Data.createLiteralLop(ValueType.INT64, String.valueOf(repCols ? hop.getDim2() : hop.getDim1()));
		}
		else {
			offset = new UnaryCP(hop.constructLops(), 
				repCols ? OpOp1.NCOL : OpOp1.NROW, DataType.SCALAR, ValueType.INT64);
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
	 * @return output size memory estimate in bytes
	 */
	protected double getOutputSize() {
		return _outputMemEstimate;
	}
	
	protected double getInputSize() {
		return getInputSize(null);
	}

	/**
	 * Get the memory estimate of inputs as the sum of input estimates in bytes.
	 * @param exclVars name of input hops to exclude from the input estimate
	 * @param injectedDefault default memory estimate (bytes) used when the memory estimate of the input is negative
	 * @return input memory estimate in bytes
	 */
	protected double getInputSize(Collection<String> exclVars, double injectedDefault){
		double sum = 0;
		int len = _input.size();
		for( int i=0; i<len; i++ ) { //for all inputs
			Hop hi = _input.get(i);
			if( exclVars != null && exclVars.contains(hi.getName()) )
				continue;
			double hmout = hi.getOutputMemEstimate(injectedDefault);
			if (hmout < 0)
				hmout = injectedDefault*(Math.max(hi.getDim1(),1) * Math.max(hi.getDim2(),1));
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

	/**
	 * Get the memory estimate of inputs as the sum of input estimates in bytes.
	 * @param exclVars name of input hops to exclude from the input estimate
	 * @return input memory estimate in bytes
	 */
	protected double getInputSize(Collection<String> exclVars) {
		return getInputSize(exclVars, OptimizerUtils.INVALID_SIZE);
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
	 *   should store its worst-case output statistics (if known) in that table.
	 * * Invocation: Intended to be called for ALL root nodes of one Hops DAG with the same
	 *   (initially empty) memo table.
	 * 
	 * @return memory estimate in bytes
	 */
	public double getMemEstimate() {
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
	public void setMemEstimate( double mem ) {
		_memEstimate = mem;
	}
	
	public void clearMemEstimate() {
		_memEstimate = OptimizerUtils.INVALID_SIZE;
	}

	public boolean isMemEstimated() {
		return (_memEstimate != OptimizerUtils.INVALID_SIZE);
	}

	//wrappers for meaningful public names to memory estimates.

	/**
	 * Get the memory estimate of inputs as the sum of input estimates in bytes.
	 * @return input memory estimate in bytes
	 */
	public double getInputMemEstimate()
	{
		return getInputSize();
	}

	/**
	 * Get the memory estimate of inputs as the sum of input estimates in bytes.
	 * @param injectedDefault default memory estimate (bytes) used when the memory estimate of the input is negative
	 * @return input memory estimate in bytes
	 */
	public double getInputMemEstimate(double injectedDefault){
		return getInputSize(null, injectedDefault);
	}

	/**
	 * Output memory estimate in bytes.
	 * @return output memory estimate in bytes
	 */
	public double getOutputMemEstimate()
	{
		return getOutputSize();
	}

	/**
	 * Output memory estimate in bytes with negative memory estimates replaced by the injected default.
	 * The injected default represents the memory estimate per output cell, hence it is multiplied by the estimated
	 * dimensions of the output of the hop.
	 * @param injectedDefault memory estimate to be returned in case the memory estimate defaults to a negative number
	 * @return output memory estimate in bytes
	 */
	public double getOutputMemEstimate(double injectedDefault)
	{
		return Math.max(getOutputMemEstimate(),injectedDefault*(Math.max(getDim1(),1) * Math.max(getDim2(),1)));
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
	public void computeMemEstimate(MemoTable memo)
	{
		DataCharacteristics wdc = null; 
		
		////////
		//Step 1) Compute hop output memory estimate (incl size inference) 
		
		switch( getDataType() )
		{
			case SCALAR: {
				//memory estimate always known
				if( getValueType()== ValueType.FP64) //default case
					_outputMemEstimate = OptimizerUtils.DOUBLE_SIZE;
				else //literalops, dataops
					_outputMemEstimate = computeOutputMemEstimate( getDim1(), getDim2(), getNnz() );
				break;
			}
			case FRAME:
			case MATRIX:
			case TENSOR:
			case LIST:
			{
				if(isCompressedOutput() && _compressedSize >= 0){
					_outputMemEstimate =  _compressedSize;
				}
				//1a) mem estimate based on exactly known dimensions and sparsity
				else if( dimsKnown(true) ) { 
					//nnz always exactly known (see dimsKnown(true))
					_outputMemEstimate = computeOutputMemEstimate(getDim1(), getDim2(), getNnz());
				}
				//1b) infer output statistics and mem estimate based on worst-case statistics
				else if( memo.hasInputStatistics(this) )
				{
					//infer the output stats
					wdc = inferOutputCharacteristics(memo);
					
					if( wdc != null && wdc.dimsKnown() ) {
						//use worst case characteristics to estimate mem
						long lnnz = wdc.nnzKnown() ? wdc.getNonZeros() : wdc.getLength();
						_outputMemEstimate = computeOutputMemEstimate(wdc.getRows(), wdc.getCols(), lnnz );
						
						//propagate worst-case estimate
						memo.memoizeStatistics(getHopID(), wdc);
					}
					else if( dimsKnown() ) {
						//nnz unknown, estimate mem as dense
						long lnnz = getLength();
						_outputMemEstimate = computeOutputMemEstimate(getDim1(), getDim2(), lnnz );
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
					long lnnz = getLength();
					_outputMemEstimate = computeOutputMemEstimate(getDim1(), getDim2(), lnnz);
				}
				//1d) fallback: unknown output size
				else {
					_outputMemEstimate = OptimizerUtils.DEFAULT_SIZE;
				}
				
				break;
			}
			default: {
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
			_processingMemEstimate = computeIntermediateMemEstimate(getDim1(), getDim2(), getNnz());
		}
		else if( wdc != null ) {
			//use worst case characteristics to estimate mem
			long lnnz = wdc.nnzKnown() ? wdc.getNonZeros() : wdc.getLength();
			_processingMemEstimate = computeIntermediateMemEstimate(wdc.getRows(), wdc.getCols(), lnnz );
		}
		else if( dimsKnown() ){
			//nnz unknown, estimate mem as dense
			long lnnz = getLength();
			_processingMemEstimate = computeIntermediateMemEstimate(getDim1(), getDim2(), lnnz);
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
	 * @return output characteristics
	 */
	protected abstract DataCharacteristics inferOutputCharacteristics(MemoTable memo);

	/**
	 * Recursively computes memory estimates for all the Hops in the DAG rooted at the 
	 * current hop pointed by <code>this</code>.
	 * 
	 * @param memo memory table
	 */
	public void refreshMemEstimates(MemoTable memo) {
		if( isVisited() )
			return;
		for( Hop h : this.getInput() )
			h.refreshMemEstimates( memo );
		computeMemEstimate( memo );
		setVisited();
	}

	/**
	 * This method determines the execution type (CP, SP) based ONLY on the 
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
			LOG.debug(s);
		}
		
		return et;
	}

	/**
	 * Checks if ExecType is federated.
	 * @return true if ExecType is federated
	 */
	public boolean isFederated(){
		return getExecType() == ExecType.FED;
	}

	public boolean someInputFederated(){
		return getInput().stream().anyMatch(Hop::hasFederatedOutput);
	}

	/**
	 * Checks if the hop is a DataOp with federated data.
	 * @return true if hop is a federated DataOp
	 */
	public boolean isFederatedDataOp(){
		return false;
	}

	public List<Hop> getParent() {
		return _parent;
	}

	public List<Hop> getInput() {
		return _input;
	}
	
	public Hop getInput(int ix) {
		return _input.get(ix);
	}
	
	public void addInput( Hop h ) {
		_input.add(h);
		h._parent.add(this);
	}
	
	public void addAllInputs( List<Hop> list ) {
		for( Hop h : list )
			addInput(h);
	}

	public int getBlocksize() {
		return _dc.getBlocksize();
	}

	public void setBlocksize(int blen) {
		_dc.setBlocksize(blen);
	}
	
	public void setNnz(long nnz){
		_dc.setNonZeros(nnz);
	}
	
	public long getNnz(){
		return _dc.getNonZeros();
	}

	public FederatedOutput getFederatedOutput(){
		return _federatedOutput;
	}

	public boolean hasFederatedOutput(){
		return _federatedOutput == FederatedOutput.FOUT;
	}

	public boolean hasLocalOutput(){
		return _federatedOutput == FederatedOutput.LOUT;
	}

	public void setUpdateType(UpdateType update){
		_updateType = update;
	}
	
	public UpdateType getUpdateType(){
		return _updateType;
	}

	public abstract Lop constructLops();

	protected final ExecType optFindExecType() {
		return optFindExecType(OptimizerUtils.ALLOW_TRANSITIVE_SPARK_EXEC_TYPE);
	}
	
	protected abstract ExecType optFindExecType(boolean transitive);
	
	public abstract String getOpString();

	@Override
	public final String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(getClass().getSimpleName());
		sb.append(" ");
		sb.append(getOpString());
		return sb.toString();
	}

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
		return (dimsKnown() && (_dc.getRows() == 1 || _dc.getCols() == 1) );
	}
	
	protected boolean areDimsBelowThreshold() {
		return (dimsKnown() && _dc.getRows() <= Hop.CPThreshold && _dc.getCols() <= Hop.CPThreshold );
	}
	
	public boolean dimsKnown() {
		return ( _dataType == DataType.SCALAR 
			|| ((_dataType==DataType.MATRIX || _dataType==DataType.FRAME || _dataType==DataType.LIST) 
				&& _dc.rowsKnown() && _dc.colsKnown()) );
	}
	
	public boolean dimsKnown(boolean includeNnz) {
		return rowsKnown() && colsKnown()
			&& (_dataType.isScalar() || ((includeNnz) ? _dc.nnzKnown() : true));
	}

	public boolean dimsKnownAny() {
		return rowsKnown() || colsKnown();
	}
	
	public boolean rowsKnown() {
		return _dataType.isScalar() || _dc.rowsKnown();
	}
	
	public boolean colsKnown() {
		return _dataType.isScalar() || _dc.colsKnown();
	}
	
	public static void resetVisitStatus( List<Hop> hops ) {
		if( hops != null )
			for( Hop hopRoot : hops )
				hopRoot.resetVisitStatus();
	}
	
	public static void resetVisitStatus( List<Hop> hops, boolean force ) {
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

	/**
	 * Get the number of rows in the Hop.
	 * @return a long.
	 */
	public long getDim1() {
		return _dc.getRows();
	}

	public void setDim1(long dim1) {
		_dc.setRows(dim1);
	}

	/**
	 * Get the number of columns in the Hop.
	 * @return a long.
	 */
	public long getDim2() {
		return _dc.getCols();
	}

	public void setDim2(long dim2) {
		_dc.setCols(dim2);
	}
	
	public long getDim(int i) {
		return _dc.getDim(i);
	}
	
	public void setDim(int i, long dim) {
		_dc.setDim(i, dim);
	}
	
	public long getLength() {
		return _dc.getLength();
	}
	
	public double getSparsity() {
		return OptimizerUtils.getSparsity(_dc);
	}
	
	public DataCharacteristics getDataCharacteristics() {
		return _dc;
	}
	
	protected void setOutputDimensions(Lop lop) {
		lop.getOutputParameters().setDimensions(
			getDim1(), getDim2(), getBlocksize(), getNnz(), getUpdateType());
	}

	protected void setMarkForLineageCaching(Lop lop) {
		//TODO: set the flag in the HOP via a rewrite
		//lop.getOutputParameters().setLineageCacheCandidate(requiresLineageCaching());
		if (!LineageCacheConfig.ReuseCacheType.isNone())
			lop.getOutputParameters().setLineageCacheCandidate(true);
	}

	protected void setOutputDimensionsIncludeCompressedSize(Lop lop) {
		lop.getOutputParameters().setDimensions(
			getDim1(), getDim2(), getBlocksize(), getNnz(), getUpdateType(), getCompressedSize());
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
	 * </ul>
	 */
	protected void setRequiresRecompileIfNecessary() {
		//NOTE: when changing these conditions, remember to update the code for
		//function recompilation in FunctionProgramBlock accordingly
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

	public static long computeSizeInformation(Hop input) {
		long ret = -1;

		try {
			long tmp = OptimizerUtils.rEvalSimpleLongExpression(input, new HashMap<Long, Long>());
			if(tmp != Long.MAX_VALUE)
				ret = tmp;
		}
		catch(Exception ex) {
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

	public void refreshRowsParameterInformation( Hop input, LocalVariableMap vars, Map<Long,Long> memo ) {
		setDim1(computeSizeInformation(input, vars, memo));
	}
	
	public void refreshColsParameterInformation( Hop input, LocalVariableMap vars ) {
		setDim2(computeSizeInformation(input, vars));
	}
	
	public void refreshColsParameterInformation( Hop input, LocalVariableMap vars, Map<Long,Long> memo ) {
		setDim2(computeSizeInformation(input, vars, memo));
	}

	public long computeSizeInformation( Hop input, LocalVariableMap vars ) {
		return computeSizeInformation(input, vars, new HashMap<Long,Long>());
	}
	
	public long computeSizeInformation( Hop input, LocalVariableMap vars, Map<Long,Long> memo )
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
	
	public static double computeBoundsInformation( Hop input, LocalVariableMap vars ) {
		return computeBoundsInformation(input, vars, new HashMap<Long, Double>());
	}
	
	public static double computeBoundsInformation( Hop input, LocalVariableMap vars, Map<Long, Double> memo ) {
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
			if( ((UnaryOp)input).getOp() == OpOp1.NROW ) {
				DataCharacteristics mc = memo.getAllInputStats(input.getInput().get(0));
				if( mc.rowsKnown() )
					ret = mc.getRows();
			}
			else if ( ((UnaryOp)input).getOp() == OpOp1.NCOL ) {
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
			if(uroot.getOp() == OpOp1.NROW)
			{
				DataCharacteristics mc = memo.getAllInputStats(uroot.getInput().get(0));
				dim = mc.getRows();
			}
			else if( uroot.getOp() == OpOp1.NCOL )
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
						case PLUS:  ret = lret + rret; break;
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
		_dc.set(that._dc);
		_updateType = that._updateType;

		//no copy of lops (regenerated)
		_parent = new ArrayList<>(_parent.size());
		_input = new ArrayList<>(_input.size());
		_lops = null;
		
		_etype = that._etype;
		_etypeForced = that._etypeForced;
		_federatedOutput = that._federatedOutput;
		_outputMemEstimate = that._outputMemEstimate;
		_memEstimate = that._memEstimate;
		_processingMemEstimate = that._processingMemEstimate;
		_requiresRecompile = that._requiresRecompile;
		_requiresReblock = that._requiresReblock;
		_requiresCheckpoint = that._requiresCheckpoint;
		_requiresCompression = that._requiresCompression;
		_requiresDeCompression = that._requiresDeCompression;
		_requiresLineageCaching = that._requiresLineageCaching;
		_compressedWorkloadTree = that._compressedWorkloadTree;
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
	
	@Override
	public void setBeginLine(int passed)    { _beginLine = passed;   }
	@Override
	public void setBeginColumn(int passed)  { _beginColumn = passed; }
	@Override
	public void setEndLine(int passed)      { _endLine = passed;   }
	@Override
	public void setEndColumn(int passed)    { _endColumn = passed; }
	@Override
	public void setFilename(String passed)  { _filename = passed; }
	@Override
	public void setText(String text)        { _text = text; }

	@Override
	public int getBeginLine()   { return _beginLine;   }
	@Override
	public int getBeginColumn() { return _beginColumn; }
	@Override
	public int getEndLine()     { return _endLine;   }
	@Override
	public int getEndColumn()   { return _endColumn; }
	@Override
	public String getFilename() { return _filename; }
	@Override
	public String getText()     { return _text; }
	
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
	protected void setLineNumbers(Lop lop) {
		lop.setAllPositions(getFilename(), getBeginLine(), getBeginColumn(), getEndLine(), getEndColumn());
	}

	protected void setMemoryAndComputeEstimates(Lop lop) {
		lop.setMemoryEstimates(getOutputMemEstimate(), getMemEstimate(),
			getIntermediateMemEstimate(), getSpBroadcastSize());
		lop.setComputeEstimate(ComputeCost.getHOPComputeCost(this));
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
