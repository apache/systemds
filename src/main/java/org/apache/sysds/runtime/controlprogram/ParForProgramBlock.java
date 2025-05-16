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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import java.util.Queue;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.ParForStatementBlock;
import org.apache.sysds.parser.ParForStatementBlock.ResultVar;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.VariableSet;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.DataPartitioner;
import org.apache.sysds.runtime.controlprogram.parfor.DataPartitionerLocal;
import org.apache.sysds.runtime.controlprogram.parfor.DataPartitionerRemoteSpark;
import org.apache.sysds.runtime.controlprogram.parfor.LocalParWorker;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.controlprogram.parfor.ParForBody;
import org.apache.sysds.runtime.controlprogram.parfor.RemoteDPParForSpark;
import org.apache.sysds.runtime.controlprogram.parfor.RemoteParForJobReturn;
import org.apache.sysds.runtime.controlprogram.parfor.RemoteParForSpark;
import org.apache.sysds.runtime.controlprogram.parfor.ResultMerge;
import org.apache.sysds.runtime.controlprogram.parfor.ResultMergeFrameLocalMemory;
import org.apache.sysds.runtime.controlprogram.parfor.ResultMergeLocalAutomatic;
import org.apache.sysds.runtime.controlprogram.parfor.ResultMergeLocalFile;
import org.apache.sysds.runtime.controlprogram.parfor.ResultMergeLocalMemory;
import org.apache.sysds.runtime.controlprogram.parfor.ResultMergeRemoteSpark;
import org.apache.sysds.runtime.controlprogram.parfor.Task;
import org.apache.sysds.runtime.controlprogram.parfor.TaskPartitioner;
import org.apache.sysds.runtime.controlprogram.parfor.TaskPartitionerFactory;
import org.apache.sysds.runtime.controlprogram.parfor.opt.OptTreeConverter;
import org.apache.sysds.runtime.controlprogram.parfor.opt.OptimizationWrapper;
import org.apache.sysds.runtime.controlprogram.parfor.opt.OptimizerRuleBased;
import org.apache.sysds.runtime.controlprogram.parfor.opt.ProgramRecompiler;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDHandler;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.instructions.cp.BooleanObject;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.IntObject;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.cp.StringObject;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.util.CollectionUtils;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.ProgramConverter;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;
import org.apache.sysds.utils.stats.ParForStatistics;
import org.apache.sysds.utils.stats.Timing;



/**
 * The ParForProgramBlock has the same execution semantics as a ForProgamBlock but executes
 * the independent iterations in parallel. See ParForStatementBlock for the loop dependency
 * analysis. At runtime level, iterations are guaranteed to be completely independent.
 * 
 * NEW FUNCTIONALITIES
 * TODO: reduction variables (operations: +=, -=, /=, *=, min, max)
 * TODO: papply(A,1:2,FUN) language construct (compiled to ParFOR) via DML function repository =&gt; modules OK, but second-order functions required
 *
 */
public class ParForProgramBlock extends ForProgramBlock {	
	protected static final Log LOG = LogFactory.getLog(ParForProgramBlock.class.getName());
	// execution modes
	public enum PExecMode {
		LOCAL,          //local (master) multi-core execution mode
		REMOTE_SPARK,   //remote (Spark cluster) execution mode
		REMOTE_SPARK_DP,//remote (Spark cluster) execution mode, fused with data partitioning
		UNSPECIFIED
	}

	// task partitioner
	public enum PTaskPartitioner {
		FIXED,          //fixed-sized task partitioner, uses tasksize 
		NAIVE,          //naive task partitioner (tasksize=1)
		STATIC,         //static task partitioner (numIterations/numThreads)
		FACTORING,      //factoring task partitioner  
		FACTORING_CMIN, //constrained factoring task partitioner, uses tasksize as min constraint
		FACTORING_CMAX, //constrained factoring task partitioner, uses tasksize as max constraint
		UNSPECIFIED
	}
	
	public enum PDataPartitionFormat {
		NONE,
		ROW_WISE,
		ROW_BLOCK_WISE,
		ROW_BLOCK_WISE_N,
		COLUMN_WISE,
		COLUMN_BLOCK_WISE,
		COLUMN_BLOCK_WISE_N,
		BLOCK_WISE_M_N;

		/**
		 * Note: Robust version of valueOf in order to return NONE without exception
		 * if misspelled or non-existing and for case-insensitivity.
		 * 
		 * @param s data partition format as string
		 * @return data partition format
		 */
		public static PDataPartitionFormat parsePDataPartitionFormat(String s) {
			if (s.equalsIgnoreCase("ROW_WISE"))
				return ROW_WISE;
			else if (s.equalsIgnoreCase("ROW_BLOCK_WISE"))
				return ROW_BLOCK_WISE;
			else if (s.equalsIgnoreCase("ROW_BLOCK_WISE_N"))
				return ROW_BLOCK_WISE_N;
			else if (s.equalsIgnoreCase("COLUMN_WISE"))
				return COLUMN_WISE;
			else if (s.equalsIgnoreCase("COLUMN_BLOCK_WISE"))
				return COLUMN_BLOCK_WISE;
			else if (s.equalsIgnoreCase("COLUMN_BLOCK_WISE_N"))
				return COLUMN_BLOCK_WISE_N;
			else if (s.equalsIgnoreCase("BLOCK_WISE_M_N"))
				return BLOCK_WISE_M_N;
			else
				return NONE;
		}
	}
	
	/**
	 * Convenience class to package PDataPartitionFormat and its parameters.
	 */
	public static class PartitionFormat implements Serializable {
		private static final long serialVersionUID = 4729309847778707801L;
		public static final PartitionFormat NONE = new PartitionFormat(PDataPartitionFormat.NONE, -1);
		public static final PartitionFormat ROW_WISE = new PartitionFormat(PDataPartitionFormat.ROW_WISE, -1);
		public static final PartitionFormat COLUMN_WISE = new PartitionFormat(PDataPartitionFormat.COLUMN_WISE, -1);
		
		public final PDataPartitionFormat _dpf;
		public final int _N;
		public PartitionFormat(PDataPartitionFormat dpf, int N) {
			_dpf = dpf;
			_N = N;
		}
		@Override
		public int hashCode() {
			return UtilFunctions.intHashCode(_dpf.ordinal(), _N);
		}
		@Override
		public boolean equals(Object o) {
			return (o instanceof PartitionFormat)
				&& _dpf == ((PartitionFormat)o)._dpf
				&& _N == ((PartitionFormat)o)._N;
		}
		@Override
		public String toString() {
			return _dpf.name()+","+_N;
		}
		public static PartitionFormat valueOf(String value) {
			String[] parts = value.split(",");
			return new PartitionFormat(PDataPartitionFormat
				.parsePDataPartitionFormat(parts[0]), Integer.parseInt(parts[1]));
		}
		public boolean isBlockwise() {
			return _dpf == PDataPartitionFormat.COLUMN_BLOCK_WISE_N 
				|| _dpf == PDataPartitionFormat.ROW_BLOCK_WISE_N;
		}
		public boolean isRowwise() {
			return _dpf == PDataPartitionFormat.ROW_WISE
				|| _dpf == PDataPartitionFormat.ROW_BLOCK_WISE
				|| _dpf == PDataPartitionFormat.ROW_BLOCK_WISE_N;
		}
		public long getNumParts(DataCharacteristics mc) {
			switch( _dpf ) {
				case ROW_WISE: return mc.getRows();
				case ROW_BLOCK_WISE: return mc.getNumRowBlocks();
				case ROW_BLOCK_WISE_N: return (long)Math.ceil((double)mc.getRows()/_N);
				case COLUMN_WISE: return mc.getCols();
				case COLUMN_BLOCK_WISE: return mc.getNumColBlocks();
				case COLUMN_BLOCK_WISE_N: return (long)Math.ceil((double)mc.getCols()/_N);
				default:
					throw new RuntimeException("Unsupported partition format: "+_dpf);
			}
		}
		public long getNumRows(DataCharacteristics mc) {
			switch( _dpf ) {
				case ROW_WISE: return 1;
				case ROW_BLOCK_WISE: return mc.getBlocksize();
				case ROW_BLOCK_WISE_N: return _N;
				case COLUMN_WISE: return mc.getRows();
				case COLUMN_BLOCK_WISE: return mc.getRows();
				case COLUMN_BLOCK_WISE_N: return mc.getRows();
				default:
					throw new RuntimeException("Unsupported partition format: "+_dpf);
			}
		}
		public long getNumColumns(DataCharacteristics mc) {
			switch( _dpf ) {
				case ROW_WISE: return mc.getCols();
				case ROW_BLOCK_WISE: return mc.getCols();
				case ROW_BLOCK_WISE_N: return mc.getCols();
				case COLUMN_WISE: return 1;
				case COLUMN_BLOCK_WISE: return mc.getBlocksize();
				case COLUMN_BLOCK_WISE_N: return _N;
				default:
					throw new RuntimeException("Unsupported partition format: "+_dpf);
			}
		}
	}
	
	public enum PDataPartitioner {
		NONE,            // no data partitioning
		LOCAL,           // local file based partition split on master node
		REMOTE_SPARK,    // remote partition split using a spark job
		UNSPECIFIED,
	}

	public enum PResultMerge {
		LOCAL_MEM,       // in-core (in-memory) result merge (output and one input at a time)
		LOCAL_FILE,      // out-of-core result merge (file format dependent)
		LOCAL_AUTOMATIC, // decides between MEM and FILE based on the size of the output matrix 
		REMOTE_SPARK,    // remote Spark parallel result merge
		UNSPECIFIED;
		public boolean isLocal() {
			return this == LOCAL_MEM 
				|| this == LOCAL_FILE
				|| this == LOCAL_AUTOMATIC;
		}
	}
	
	//optimizer
	public enum POptMode{
		NONE,            //no optimization, use defaults and specified parameters
		RULEBASED,       //rule-based rewritings with memory constraints 
		CONSTRAINED,     //same as rule-based but with given params as constraints
		HEURISTIC,       //same as rule-based but with time-based cost estimates
	}
	
	// internal parameters
	public static final boolean OPTIMIZE                    = true; // run all automatic optimizations on top-level parfor
	public static final boolean USE_PB_CACHE                = false; // reuse copied program blocks whenever possible, not there can be issues related to recompile
	public static final boolean USE_RANGE_TASKS_IF_USEFUL   = true; // use range tasks whenever size>3, false, otherwise wrong split order in remote 
	public static final boolean USE_STREAMING_TASK_CREATION = true; // start working while still creating tasks, prevents blocking due to too small task queue
	public static final boolean ALLOW_NESTED_PARALLELISM    = true; // if not, transparently change parfor to for on program conversions (local,remote)
	public static final boolean CONVERT_NESTED_REMOTE_PARFOR = true; //convert parfor to for in remote parfor
	public static final boolean USE_PARALLEL_RESULT_MERGE   = false; // if result merge is run in parallel or serial 
	public static final boolean USE_PARALLEL_RESULT_MERGE_REMOTE = true; // if remote result merge should be run in parallel for multiple result vars
	public static final boolean CREATE_UNSCOPED_RESULTVARS  = true;
	public static       boolean ALLOW_REUSE_PARTITION_VARS  = true; //reuse partition input matrices, applied only if read-only in surrounding loops
	public static final int     WRITE_REPLICATION_FACTOR    = 1;
	public static       int     MAX_RETRYS_ON_ERROR         = 1;
	public static final boolean FORCE_CP_ON_REMOTE_SPARK    = true; // compile body to CP if exec type forced to Spark
	public static final boolean LIVEVAR_AWARE_EXPORT        = true; // export only read variables according to live variable analysis
	public static final boolean RESET_RECOMPILATION_FLAGs   = true;
	public static       boolean ALLOW_BROADCAST_INPUTS      = true; // enables to broadcast inputs for remote_spark
	public static final boolean COPY_EVAL_FUNCTIONS         = true; // copy eval functions similar to normal parfor functions
	
	public static final String PARFOR_FNAME_PREFIX          = "/parfor/"; 
	public static final String PARFOR_MR_TASKS_TMP_FNAME    = PARFOR_FNAME_PREFIX + "%ID%_MR_taskfile"; 
	public static final String PARFOR_MR_RESULT_TMP_FNAME   = PARFOR_FNAME_PREFIX + "%ID%_MR_results"; 
	public static final String PARFOR_MR_RESULTMERGE_FNAME  = PARFOR_FNAME_PREFIX + "%ID%_resultmerge%VAR%"; 
	public static final String PARFOR_DATAPARTITIONS_FNAME  = PARFOR_FNAME_PREFIX + "%ID%_datapartitions%VAR%"; 
	
	public static final String PARFOR_COUNTER_GROUP_NAME    = "SystemDS ParFOR Counters";

	// static ID generator sequences
	private final static IDSequence _pfIDSeq = new IDSequence();
	private final static IDSequence _pwIDSeq = new IDSequence();
	
	// runtime parameters
	protected final HashMap<String,String> _params;
	protected final Level _optLogLevel;
	protected int _numThreads = -1;
	protected boolean _fixedDOP = false; //guard for numThreads
	protected long _taskSize = -1;
	protected PTaskPartitioner _taskPartitioner;
	protected PDataPartitioner _dataPartitioner;
	protected PResultMerge _resultMerge;
	protected PExecMode _execMode;
	protected POptMode _optMode;
	
	//specifics used for optimization
	protected long _numIterations = -1;
	
	//specifics used for data partitioning
	protected LocalVariableMap _variablesDPOriginal;
	protected LocalVariableMap _variablesDPReuse;
	protected String _colocatedDPMatrix = null;
	protected boolean _tSparseCol = false;
	protected int _replicationDP = WRITE_REPLICATION_FACTOR;
	protected int _replicationExport = -1;
	//specifics used for result partitioning
	protected boolean _jvmReuse = true;
	//specifics used for recompilation 
	protected double _oldMemoryBudget = -1;
	protected double _recompileMemoryBudget = -1;
	//specifics for caching
	protected boolean _enableCPCaching = true;
	protected boolean _enableRuntimePiggybacking = false;
	//specifics for spark 
	protected Collection<String> _variablesRP = null;
	protected Collection<String> _variablesECache = null;
	
	// program block meta data
	protected final ArrayList<ResultVar> _resultVars;
	protected final IDSequence _resultVarsIDSeq;
	protected final IDSequence _dpVarsIDSeq;
	protected final boolean _hasFunctions;
	
	protected long _ID = -1;
	protected int _IDPrefix = -1;
	protected int _numRuns = -1;
	
	// local parworker data
	protected HashMap<Long,ArrayList<ProgramBlock>> _pbcache = null;
	protected long[] _pwIDs = null;
	
	public ParForProgramBlock(Program prog, String iterPredVar, HashMap<String,String> params, ArrayList<ResultVar> resultVars) {
		this( -1, prog, iterPredVar, params, resultVars);
	}
	
	/**
	 * ParForProgramBlock constructor. It reads the specified parameter settings, where defaults for non-specified parameters
	 * have been set in ParForStatementBlock.validate(). Furthermore, it generates the IDs for the ParWorkers.
	 * 
	 * @param ID parfor program block id
	 * @param prog runtime program
	 * @param iterPredVar ?
	 * @param params map of parameters
	 * @param resultVars list of result variable names
	 */
	public ParForProgramBlock(int ID, Program prog, String iterPredVar, HashMap<String,String> params, ArrayList<ResultVar> resultVars) 
	{
		super(prog, iterPredVar);
		
		//ID generation and setting 
		setParForProgramBlockIDs( ID );
		_resultVars = resultVars;
		_resultVarsIDSeq = new IDSequence();
		_dpVarsIDSeq = new IDSequence();
		
		//parse and use internal parameters (already set to default if not specified)
		_params = params;
		try {
			_numThreads      = Integer.parseInt( getParForParam(ParForStatementBlock.PAR) );
			_taskPartitioner = PTaskPartitioner.valueOf( getParForParam(ParForStatementBlock.TASK_PARTITIONER) );
			_taskSize        = Integer.parseInt( getParForParam(ParForStatementBlock.TASK_SIZE) );
			_dataPartitioner = PDataPartitioner.valueOf( getParForParam(ParForStatementBlock.DATA_PARTITIONER) );
			_resultMerge     = PResultMerge.valueOf( getParForParam(ParForStatementBlock.RESULT_MERGE) );
			_execMode        = PExecMode.valueOf( getParForParam(ParForStatementBlock.EXEC_MODE) );
			_optMode         = POptMode.valueOf( getParForParam(ParForStatementBlock.OPT_MODE) );
			_optLogLevel     = Level.toLevel( getParForParam(ParForStatementBlock.OPT_LOG) );
		}
		catch(Exception ex) {
			throw new RuntimeException("Error parsing specified ParFOR parameters.",ex);
		}
		
		//reset the internal opt mode if optimization globally disabled.
		if( !OPTIMIZE )
			_optMode = POptMode.NONE;
		
		_variablesDPOriginal = new LocalVariableMap();
		_variablesDPReuse = new LocalVariableMap();
		
		//create IDs for all parworkers
		if( _execMode == PExecMode.LOCAL /*&& _optMode==POptMode.NONE*/ )
			setLocalParWorkerIDs();
	
		//initialize program block cache if necessary
		if( USE_PB_CACHE ) 
			_pbcache = new HashMap<>();
		
		//created profiling report after parfor exec
		_numRuns = 0;
		
		//materialized meta data (reused for all invocations)
		_hasFunctions = ProgramRecompiler.containsAtLeastOneFunction(this);
		
		LOG.trace("PARFOR: ParForProgramBlock created with mode = "+_execMode+", optmode = "+_optMode+", numThreads = "+_numThreads);
	}
	
	public static void resetWorkerIDs() {
		_pwIDSeq.reset();
	}
	
	public long getID() {
		return _ID;
	}
	
	public PExecMode getExecMode() {
		return _execMode;
	}
	
	public HashMap<String,String> getParForParams() {
		return _params;
	}
	
	public String getParForParam(String key) {
		String tmp = getParForParams().get(key);
		return (tmp == null) ? null :
			UtilFunctions.unquote(tmp).toUpperCase();
	}

	public ArrayList<ResultVar> getResultVariables() {
		return _resultVars;
	}
	
	public void disableOptimization() {
		_optMode = POptMode.NONE;
	}
	
	public POptMode getOptimizationMode() {
		return _optMode;
	}
	
	public void setOptimizationMode(POptMode mode) {
		_optMode = mode;
	}
	
	public int getDegreeOfParallelism() {
		return _numThreads;
	}
	
	public void setDegreeOfParallelism(int k) {
		_numThreads = k;
		_params.put(ParForStatementBlock.PAR, String.valueOf(_numThreads)); //kept up-to-date for copies
		setLocalParWorkerIDs();
	}
	
	public boolean isDegreeOfParallelismFixed() {
		return _fixedDOP;
	}
	
	public void setDegreeOfParallelismFixed(boolean flag) {
		_fixedDOP = flag;
	}

	public void setCPCaching(boolean flag) {
		_enableCPCaching = flag;
	}
	
	public void setRuntimePiggybacking(boolean flag) {
		_enableRuntimePiggybacking = flag;
	}
	
	public void setExecMode( PExecMode mode ) {
		_execMode = mode;
		_params.put(ParForStatementBlock.EXEC_MODE, String.valueOf(_execMode)); //kept up-to-date for copies
	}
	
	public void setTaskPartitioner( PTaskPartitioner partitioner ) {
		_taskPartitioner = partitioner;
		_params.put(ParForStatementBlock.TASK_PARTITIONER, String.valueOf(_taskPartitioner)); //kept up-to-date for copies
	}
	
	public void setTaskSize( long tasksize ) {
		_taskSize = tasksize;
		_params.put(ParForStatementBlock.TASK_SIZE, String.valueOf(_taskSize)); //kept up-to-date for copies
	}
	
	public void setDataPartitioner(PDataPartitioner partitioner)  {
		_dataPartitioner = partitioner;
		_params.put(ParForStatementBlock.DATA_PARTITIONER, String.valueOf(_dataPartitioner)); //kept up-to-date for copies
	}
	
	public void enableColocatedPartitionedMatrix( String varname ) {
		//only called from optimizer
		_colocatedDPMatrix = varname;
	}
	
	public void setTransposeSparseColumnVector( boolean flag ) {
		_tSparseCol = flag;
	}
	
	public void setPartitionReplicationFactor( int rep ) {
		//only called from optimizer
		_replicationDP = rep;
	}
	
	public void setExportReplicationFactor( int rep ) {
		//only called from optimizer
		_replicationExport = rep;
	}
	
	public void disableJVMReuse() {
		//only called from optimizer
		_jvmReuse = false;
	}
	
	public void setResultMerge(PResultMerge merge) {
		_resultMerge = merge;
		_params.put(ParForStatementBlock.RESULT_MERGE, String.valueOf(_resultMerge)); //kept up-to-date for copies
	}
	
	public void setRecompileMemoryBudget( double localMem ) {
		_recompileMemoryBudget = localMem;
	}
	
	public void setSparkRepartitionVariables(Collection<String> vars) {
		_variablesRP = vars;
	}
	
	public Collection<String> getSparkRepartitionVariables() {
		return _variablesRP;
	}
	
	public void setSparkEagerCacheVariables(Collection<String> vars) {
		_variablesECache = vars;
	}
	
	public long getNumIterations() {
		return _numIterations;
	}
	
	public boolean hasFunctions() {
		return _hasFunctions;
	}
	
	@Override
	public void execute(ExecutionContext ec)
	{
		ParForStatementBlock sb = (ParForStatementBlock)getStatementBlock();
		
		// evaluate from, to, incr only once (assumption: known at for entry)
		ScalarObject from0 = executePredicateInstructions(1, _fromInstructions, ec, false);
		ScalarObject to0   = executePredicateInstructions(2, _toInstructions, ec, false);
		ScalarObject incr0 = (_incrementInstructions == null || _incrementInstructions.isEmpty()) ? 
			new IntObject((from0.getLongValue()<=to0.getLongValue()) ? 1 : -1) :
			executePredicateInstructions( 3, _incrementInstructions, ec, false);
		
		if ( incr0.getLongValue() == 0 ) //would produce infinite loop
			throw new DMLRuntimeException(this.printBlockErrorLocation() + "Expression for increment "
				+ "of variable '" + _iterPredVar + "' must evaluate to a non-zero value.");
		
		//early exit on num iterations (e.g., for invalid loop bounds)
		_numIterations = UtilFunctions.getSeqLength( 
			from0.getDoubleValue(), to0.getDoubleValue(), incr0.getDoubleValue(), false);
		// avoid unnecessary optimization/initialization, and issue with simplification rewrites
		// (e.g., rewriting leftindexing into a result variable, to assignment)
		if( _numIterations <= 0 )
			return; 
		if( _numIterations == 1 ) {
			//fallback to basic for loop.
			super.execute(ec);
			return;
		}
		
		IntObject from = new IntObject(from0.getLongValue());
		IntObject to = new IntObject(to0.getLongValue());
		IntObject incr = new IntObject(incr0.getLongValue());
		
		///////
		//OPTIMIZATION of ParFOR body (incl all child parfor PBs)
		///////
		if( _optMode != POptMode.NONE ) {
			OptimizationWrapper.optimize(_optMode, sb, this, ec, _numRuns); //core optimize
		}

		///////
		//DATA PARTITIONING of read-only parent variables of type (matrix,unpartitioned)
		///////
		
		//partitioning on demand (note: for fused data partitioning and execute the optimizer set 
		//the data partitioner to NONE in order to prevent any side effects)
		handleDataPartitioning( ec ); 
	
		//repartitioning of variables for spark cpmm/zipmm in order prevent unnecessary shuffle
		handleSparkRepartitioning( ec );
		
		//eager rdd caching of variables for spark in order prevent read/write contention
		handleSparkEagerCaching( ec );
		
		// initialize iter var to from value
		IntObject iterVar = new IntObject(from.getLongValue());
		
		///////
		//begin PARALLEL EXECUTION of (PAR)FOR body
		///////
		LOG.trace("EXECUTE PARFOR ID = "+_ID+" with mode = "+_execMode+", numThreads = "+_numThreads+", taskpartitioner = "+_taskPartitioner);
		
		//preserve shared input/result variables of cleanup
		ArrayList<String> varList = ec.getVarList();
		Queue<Boolean> varState = ec.pinVariables(varList);
		
		try 
		{
			switch( _execMode )
			{
				case LOCAL: //create parworkers as local threads
					executeLocalParFor(ec, from, to, incr);
					break;
				
				case REMOTE_SPARK: // create parworkers as Spark tasks (one job per parfor)
					executeRemoteSparkParFor(ec, from, to, incr);
					break;
				
				case REMOTE_SPARK_DP: // create parworkers as Spark tasks (one job per parfor)
					executeRemoteSparkParForDP(ec, from, to, incr);
					break;
				
				default:
					throw new DMLRuntimeException("Undefined execution mode: '"+_execMode+"'.");
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException("PARFOR: Failed to execute loop in parallel.",ex);
		}

		//reset state of shared input/result variables 
		ec.unpinVariables(varList, varState);
		
		//cleanup unpinned shared variables
		cleanupSharedVariables(ec, varState);
		
		//set iteration var to TO value (+ increment) for FOR equivalence
		iterVar = new IntObject(to.getLongValue()); //consistent with for
		ec.setVariable(_iterPredVar, iterVar);
		
		//ensure that subsequent program blocks never see partitioned data (invalid plans!)
		//we can replace those variables, because partitioning only applied for read-only matrices
		for( String var : _variablesDPOriginal.keySet() ) {
			//cleanup partitioned matrix (if not reused)
			if( !_variablesDPReuse.keySet().contains(var) )
				VariableCPInstruction.processRmvarInstruction(ec, var); 
			//reset to original matrix
			MatrixObject mo = (MatrixObject) _variablesDPOriginal.get( var );
			ec.setVariable(var, mo); 
		}
		
		//execute exit instructions
		executeExitInstructions("parfor", ec);
		
		///////
		//end PARALLEL EXECUTION of (PAR)FOR body
		///////
	
		//print profiling report (only if top-level parfor because otherwise in parallel context)
		_numRuns ++;
		
		//reset flags/modifications made by optimizer
		//TODO reset of hop parallelism constraint (e.g., ba+*)
		for( String dpvar : _variablesDPOriginal.keySet() ) //release forced exectypes
			ProgramRecompiler.rFindAndRecompileIndexingHOP(sb, this, dpvar, ec, false);
		 //release forced exectypes for fused dp/exec
		if( _execMode == PExecMode.REMOTE_SPARK_DP )
			ProgramRecompiler.rFindAndRecompileIndexingHOP(sb, this, _colocatedDPMatrix, ec, false); 
		resetOptimizerFlags(); //after release, deletes dp_varnames
	}


	/**
	 * Executes the parfor locally, i.e., the parfor is realized with numThreads local threads that drive execution.
	 * This execution mode allows for arbitrary nested local parallelism and nested invocations of MR jobs. See
	 * below for details of the realization.
	 * 
	 * @param ec execution context
	 * @param from ?
	 * @param to ?
	 * @param incr ?
	 * @throws InterruptedException if InterruptedException occurs
	 */
	private void executeLocalParFor( ExecutionContext ec, IntObject from, IntObject to, IntObject incr )
		throws InterruptedException
	{
		if (DMLScript.USE_ACCELERATOR) {
			_numThreads = Math.min(_numThreads, ec.getNumGPUContexts());
		}

		LOG.trace("Local Par For (multi-threaded) with degree of parallelism : " + _numThreads);
		/* Step 1) init parallel workers, task queue and threads
		 *         start threads (from now on waiting for tasks)
		 * Step 2) create tasks
		 *         put tasks into queue
		 *         mark end of task input stream
		 * Step 3) join all threads (wait for finished work)
		 * Step 4) collect results from each parallel worker
		 */

		Timing time = new Timing(true);

		int numExecutedTasks = 0;
		int numExecutedIterations = 0;
		
		//restrict recompilation to thread local memory
		setMemoryBudget();
		
		final LocalTaskQueue<Task> queue = new LocalTaskQueue<>();
		final Thread[] threads         = new Thread[_numThreads];
		final LocalParWorker[] workers = new LocalParWorker[_numThreads];
		try
		{
			// Step 1) create task queue and init workers in parallel
			// (including preparation of update-in-place variables)
			IntStream.range(0, _numThreads).forEach(i -> {
				workers[i] = createParallelWorker( _pwIDs[i], queue, ec, i);
				threads[i] = new Thread( workers[i] , "PARFOR");
				threads[i].setPriority(Thread.MAX_PRIORITY);
			});
			
			// start threads (from now on waiting for tasks)
			for( Thread thread : threads )
				thread.start();
			
			//maintain statistics
			long tinit = (long) time.stop();
			if( DMLScript.STATISTICS )
				ParForStatistics.incrementInitTime(tinit);
			
			// Step 2) create tasks 
			TaskPartitioner partitioner = createTaskPartitioner(from, to, incr);
			long numIterations = partitioner.getNumIterations();
			long numCreatedTasks = -1;
			if( USE_STREAMING_TASK_CREATION )
			{
				//put tasks into queue (parworker start work on first tasks while creating tasks) 
				numCreatedTasks = partitioner.createTasks(queue);
			}
			else
			{
				List<Task> tasks = partitioner.createTasks();
				numCreatedTasks = tasks.size();
				
				// put tasks into queue
				for( Task t : tasks )
					queue.enqueueTask( t );
				
				// mark end of task input stream
				queue.closeInput();
			}
			
			// Step 3) join all threads (wait for finished work)
			LineageCacheConfig.setReuseLineageTraces(false); //disable lineage trace reuse
			for( Thread thread : threads )
				thread.join();

			if (DMLScript.USE_ACCELERATOR) {
				for(LocalParWorker worker : workers) {
					LOG.trace("The worker of GPU " + worker.getExecutionContext().getGPUContext(0).toString() + " has executed " + worker.getExecutedTasks() + " tasks.");
				}
			}
			
			// Step 4) collecting results from each parallel worker
			//obtain results and cleanup other intermediates before result merge
			LocalVariableMap [] localVariables = new LocalVariableMap [_numThreads]; 
			for( int i=0; i<_numThreads; i++ ) {
				localVariables[i] = workers[i].getVariables();
				localVariables[i].removeAllNotIn(_resultVars.stream()
					.map(v -> v._name).collect(Collectors.toSet()));
				numExecutedTasks += workers[i].getExecutedTasks();
				numExecutedIterations += workers[i].getExecutedIterations();
			}
			// lineage maintenance (but filter out lineage DAGs of workers that 
			// did not execute a single tasks as this would corrupt the merge)
			Lineage [] lineages = Arrays.stream(workers)
				.filter(w -> w.getExecutedTasks() >= 1)
				.map(w -> w.getExecutionContext().getLineage())
				.toArray(Lineage[]::new);
			mergeLineage(ec, lineages);
			//LineageCacheConfig.setReuseLineageTraces(true);

			//consolidate results into global symbol table
			consolidateAndCheckResults( ec, numIterations, numCreatedTasks,
				numExecutedIterations, numExecutedTasks, localVariables );
			
			// Step 5) cleanup local parworkers (e.g., remove created functions)
			for( int i=0; i<_numThreads; i++ ) {
				Collection<String> fnNames = workers[i].getFunctionNames();
				if( fnNames!=null ) 
					fnNames.stream().map(fn -> DMLProgram.splitFunctionKey(fn))
						.forEach(p -> _prog.removeFunctionProgramBlock(p[0], p[1]));
				// also cleanup worker-specific functions created via eval on-demand loading
				workers[i].getExecutionContext().getTmpParforFunctions().stream()
					.map(fn -> DMLProgram.splitFunctionKey(fn))
					.forEach(p -> {
						_prog.getDMLProg().removeFunctionStatementBlock(p[0], p[1]);
						_prog.removeFunctionProgramBlock(p[0], p[1]);
					});
			}

			// Frees up the GPUContexts used in the threaded Parfor and sets
			// the main thread to use the GPUContext
			if (DMLScript.USE_ACCELERATOR) {
				ec.getGPUContext(0).initializeThread();
			}
		}
		finally {
			//remove thread-local memory budget (reset to original budget)
			//(in finally to prevent error side effects for multiple scripts in one jvm)
			resetMemoryBudget();

			if(threads != null) 
				for(Thread t : threads) 
					CommonThreadPool.shutdownAsyncPools(t);
		}
	}

	private void executeRemoteSparkParFor(ExecutionContext ec, IntObject from, IntObject to, IntObject incr)
	{
		// Step 0) check and compile to CP (if forced remote parfor)
		boolean flagForced = false;
		if( FORCE_CP_ON_REMOTE_SPARK && _execMode==PExecMode.REMOTE_SPARK ) {
			//tid = 0  because replaced in remote parworker
			flagForced = checkSparkAndRecompileToCP(0);
		}
		
		// Step 1) init parallel workers (serialize PBs)
		// NOTES: each mapper changes filenames with regard to his ID as we submit a single 
		// job, cannot reuse serialized string, since variables are serialized as well.
		ParForBody body = new ParForBody(_childBlocks, _resultVars, ec);
		HashMap<String, byte[]> clsMap = new HashMap<>();
		String program = ProgramConverter.serializeParForBody(body, clsMap);
		
		// Step 2) create tasks 
		TaskPartitioner partitioner = createTaskPartitioner(from, to, incr);
		long numIterations = partitioner.getNumIterations();
		
		//sequentially create tasks as input to parfor job
		List<Task> tasks = partitioner.createTasks();
		long numCreatedTasks = tasks.size();
		
		//handle broadcast / export of inputs
		Set<String> brVars = getBroadcastVariables(ec, _resultVars);
		exportMatricesToHDFS(ec, brVars);
		
		// Step 3) submit Spark parfor job (no lazy evaluation, since collect on result)
		LineageCacheConfig.setReuseLineageTraces(false); //disable lineage trace reuse
		boolean topLevelPF = OptimizerUtils.isTopLevelParFor();
		RemoteParForJobReturn ret = RemoteParForSpark.runJob(_ID, program,
			clsMap, tasks, ec, brVars, _resultVars, _enableCPCaching, _numThreads, topLevelPF);
		
		// Step 4) collecting results from each parallel worker
		int numExecutedTasks = ret.getNumExecutedTasks();
		int numExecutedIterations = ret.getNumExecutedIterations();
		
		//lineage maintenance
		mergeLineage(ec, ret.getLineages());
		//LineageCacheConfig.setReuseLineageTraces(true);
		// TODO: remove duplicate lineage items in ec.getLineage()
		
		//consolidate results into global symbol table
		consolidateAndCheckResults( ec, numIterations, numCreatedTasks,
			numExecutedIterations , numExecutedTasks, ret.getVariables() );
		if( flagForced ) //see step 0
			releaseForcedRecompile(0);
	}
	
	private void executeRemoteSparkParForDP( ExecutionContext ec, IntObject from, IntObject to, IntObject incr ) {
		// Step 0) check and compile to CP (if forced remote parfor)
		boolean flagForced = checkSparkAndRecompileToCP(0);
		
		// Step 1) prepare partitioned input matrix (needs to happen before serializing the program)
		ParForStatementBlock sb = (ParForStatementBlock) getStatementBlock();
		MatrixObject inputMatrix = ec.getMatrixObject(_colocatedDPMatrix );
		PartitionFormat inputDPF = sb.determineDataPartitionFormat( _colocatedDPMatrix );
		inputMatrix.setPartitioned(inputDPF._dpf, inputDPF._N); //mark matrix var as partitioned
		
		// Step 2) init parallel workers (serialize PBs)
		// NOTES: each mapper changes filenames with regard to his ID as we submit a single
		// job, cannot reuse serialized string, since variables are serialized as well.
		ParForBody body = new ParForBody( _childBlocks, _resultVars, ec );
		HashMap<String, byte[]> clsMap = new HashMap<>(); 
		String program = ProgramConverter.serializeParForBody( body, clsMap );
		
		// Step 3) create tasks 
		TaskPartitioner partitioner = createTaskPartitioner(from, to, incr);
		String resultFile = constructResultFileName();
		long numIterations = partitioner.getNumIterations();
		long numCreatedTasks = numIterations;//partitioner.createTasks().size();
		
		//write matrices to HDFS, except DP matrix which is the input to the RemoteDPParForSpark job
		exportMatricesToHDFS(ec, CollectionUtils.asSet(_colocatedDPMatrix)); //incl colocated
		
		// Step 4) submit MR job (wait for finished work)
		RemoteParForJobReturn ret = RemoteDPParForSpark.runJob(
			_ID, _iterPredVar, _colocatedDPMatrix, program, clsMap, resultFile, inputMatrix,
			ec, inputDPF, FileFormat.BINARY, _tSparseCol, _enableCPCaching, _numThreads);

		// Step 5) collecting results from each parallel worker
		int numExecutedTasks = ret.getNumExecutedTasks();
		int numExecutedIterations = ret.getNumExecutedIterations();
		
		//consolidate results into global symbol table
		consolidateAndCheckResults( ec, numIterations, numCreatedTasks,
			numExecutedIterations, numExecutedTasks, ret.getVariables() );
		
		if( flagForced ) //see step 0
			releaseForcedRecompile(0);
		inputMatrix.unsetPartitioned();
	}

	private void handleDataPartitioning( ExecutionContext ec ) 
	{
		PDataPartitioner dataPartitioner = _dataPartitioner;
		if( dataPartitioner != PDataPartitioner.NONE )
		{
			ParForStatementBlock sb = (ParForStatementBlock) getStatementBlock();
			if( sb == null )
				throw new DMLRuntimeException("ParFor statement block required for reasoning about data partitioning.");
			
			for( String var : sb.getReadOnlyParentMatrixVars() )
			{
				Data dat = ec.getVariable(var);
				//skip non-existing input matrices (which are due to unknown sizes marked for
				//partitioning but typically related branches are never executed)
				if( dat instanceof MatrixObject )
				{
					MatrixObject moVar = (MatrixObject) dat; //unpartitioned input
					
					PartitionFormat dpf = sb.determineDataPartitionFormat( var );
					LOG.trace("PARFOR ID = "+_ID+", Partitioning read-only input variable "
						+ var + " (format="+dpf+", mode="+_dataPartitioner+")");
					
					if( dpf != PartitionFormat.NONE )
					{
						if( dataPartitioner != PDataPartitioner.REMOTE_SPARK && dpf.isBlockwise() ) {
							LOG.warn("PARFOR ID = "+_ID+", Switching data partitioner from " + dataPartitioner + 
								" to " + PDataPartitioner.REMOTE_SPARK.name()+" for blockwise-n partitioning.");
							dataPartitioner = PDataPartitioner.REMOTE_SPARK;
						}
						
						Timing ltime = new Timing(true);
						
						//input data partitioning (reuse if possible)
						Data dpdatNew = _variablesDPReuse.get(var);
						if( dpdatNew == null ) //no reuse opportunity
						{
							DataPartitioner dp = createDataPartitioner( dpf, dataPartitioner, ec );
							//disable binary cell for sparse if consumed by MR jobs
							if(    !OptimizerRuleBased.allowsBinaryCellPartitions(moVar, dpf )
								|| OptimizerUtils.isSparkExecutionMode() ) //TODO support for binarycell
							{
								dp.disableBinaryCell();
							}
							MatrixObject moVarNew = dp.createPartitionedMatrixObject(moVar, constructDataPartitionsFileName());
							dpdatNew = moVarNew;
							
							//skip remaining partitioning logic if not partitioned (e.g., too small)
							if( moVar == moVarNew ) 
								continue; //skip to next
						}
						ec.setVariable(var, dpdatNew);
						
						//recompile parfor body program
						ProgramRecompiler.rFindAndRecompileIndexingHOP(sb,this,var,ec,true);
						
						//store original and partitioned matrix (for reuse if applicable)
						_variablesDPOriginal.put(var, moVar);
						if( ALLOW_REUSE_PARTITION_VARS 
							&& ProgramRecompiler.isApplicableForReuseVariable(sb.getDMLProg(), sb, var) ) {
							_variablesDPReuse.put(var, dpdatNew);
						}
						
						LOG.trace("Partitioning and recompilation done in "+ltime.stop()+"ms");
					}
				}
			}
		}
	}

	private void handleSparkRepartitioning( ExecutionContext ec ) {
		if( OptimizerUtils.isSparkExecutionMode() &&
			_variablesRP != null && !_variablesRP.isEmpty() ) {
			SparkExecutionContext sec = (SparkExecutionContext) ec;
			for( String var : _variablesRP )
				sec.repartitionAndCacheMatrixObject(var);
		}
	}

	private void handleSparkEagerCaching( ExecutionContext ec ) {
		if( OptimizerUtils.isSparkExecutionMode() &&
			_variablesECache != null && !_variablesECache.isEmpty() ) {
			SparkExecutionContext sec = (SparkExecutionContext) ec;
			for( String var : _variablesECache )
				sec.cacheMatrixObject(var);
		}
	}
	
	/**
	 * Cleanup result variables of parallel workers after result merge.
	 * 
	 * @param ec execution context
	 * @param out output matrix
	 * @param in array of input matrix objects
	 */
	private static void cleanWorkerResultVariables(ExecutionContext ec, CacheableData<?> out, CacheableData<?>[] in, boolean parallel) {
		//check for empty inputs (no iterations executed)
		Stream<CacheableData<?>> results = Arrays.stream(in).filter(m -> m!=null && m!=out);
		//perform cleanup (parallel to mitigate file deletion bottlenecks)
		(parallel ? results.parallel() : results)
			.forEach(ec::cleanupCacheableData);
	}
	
	/**
	 * Create empty matrix objects and scalars for all unscoped vars 
	 * (created within the parfor).
	 * 
	 * NOTE: parfor gives no guarantees on the values of those objects - hence
	 * we return -1 for sclars and empty matrix objects.
	 * 
	 * @param out local variable map
	 * @param sb statement block
	 */
	private static void createEmptyUnscopedVariables( LocalVariableMap out, StatementBlock sb ) 
	{
		VariableSet updated = sb.variablesUpdated();
		VariableSet livein = sb.liveIn();
		
		//for all vars IN <updated> AND NOT IN <livein>
		for( String var : updated.getVariableNames() )
			if( !livein.containsVariable(var) )
			{
				//create empty output
				DataIdentifier dat = updated.getVariable(var);
				DataType datatype = dat.getDataType();
				ValueType valuetype = dat.getValueType();
				Data dataObj = null;
				switch( datatype )
				{
					case SCALAR:
						switch( valuetype ) {
							case BOOLEAN: dataObj = new BooleanObject(false); break;
							case INT64:     dataObj = new IntObject(-1);        break;
							case FP64:  dataObj = new DoubleObject(-1d);    break;
							case STRING:  dataObj = new StringObject("-1");   break;
							default:
								throw new DMLRuntimeException("Value type not supported: "+valuetype);
						}
						break;
					case MATRIX:
					case FRAME:
						//currently we do not create any unscoped matrix or frame outputs
						//because metadata (e.g., outputinfo) not known at this place.
						break;
					case LIST:
						dataObj = new ListObject(Collections.emptyList());
						break;
					case UNKNOWN:
						break;
					default:
						throw new DMLRuntimeException("Data type not supported: "+datatype);
				}
				
				if( dataObj != null )
					out.put(var, dataObj);
			}
	}
	
	private static Set<String> getBroadcastVariables(ExecutionContext ec, List<ResultVar> resultVars) {
		if( !ALLOW_BROADCAST_INPUTS )
			return new HashSet<>();
		LocalVariableMap inputs = ec.getVariables();
		// exclude the result variables
		Set<String> retVars = resultVars.stream()
			.map(v -> v._name).collect(Collectors.toSet());
		Set<String> brVars = inputs.keySet().stream()
			.filter(v -> !retVars.contains(v))
			.filter(v -> ec.getVariable(v).getDataType().isMatrixOrFrame())
			.filter(v -> OptimizerUtils.estimateSize(ec.getDataCharacteristics(v))< 2.14e9)
			.collect(Collectors.toSet());
		return brVars;
	}

	private void exportMatricesToHDFS(ExecutionContext ec, Set<String> excludeNames)  {
		ParForStatementBlock sb = (ParForStatementBlock)getStatementBlock();
		
		if( LIVEVAR_AWARE_EXPORT && sb != null ) {
			//optimization to prevent unnecessary export of matrices
			//export only variables that are read in the body
			VariableSet varsRead = sb.variablesRead();
			for (String key : ec.getVariables().keySet() ) {
				if( varsRead.containsVariable(key) && !excludeNames.contains(key) )
					exportDataToHDFS(ec.getVariable(key), key);
			}
		}
		else {
			//export all matrices in symbol table
			for (String key : ec.getVariables().keySet() ) {
				if( !excludeNames.contains(key) )
					exportDataToHDFS(ec.getVariable(key), key);
			}
		}
	}
	
	private void exportDataToHDFS(Data data, String key) {
		if( data.getDataType().isMatrixOrFrame() )
			((CacheableData<?>)data).exportData(_replicationExport);
		if( data.getDataType().isList() ) {
			for( Data ldata : ((ListObject)data).getData() )
				exportDataToHDFS(ldata, key);
		}
	}

	private void cleanupSharedVariables( ExecutionContext ec, Queue<Boolean> varState ) {
		//TODO needs as precondition a systematic treatment of persistent read information.
	}
	
	/**
	 * Creates a new or partially recycled instance of a parallel worker. Therefore the symbol table, and child
	 * program blocks are deep copied. Note that entries of the symbol table are not deep copied because they are replaced 
	 * anyway on the next write. In case of recycling the deep copies of program blocks are recycled from previous 
	 * executions of this parfor.
	 * 
	 * @param pwID parworker id
	 * @param queue task queue
	 * @param ec execution context
	 * @param index the index of the worker
	 * @return local parworker
	 */
	private LocalParWorker createParallelWorker(long pwID, LocalTaskQueue<Task> queue, ExecutionContext ec, int index)
	{
		LocalParWorker pw = null; 
		
		try
		{
			//create deep copies of required elements child blocks
			ArrayList<ProgramBlock> cpChildBlocks = null;
			HashSet<String> fnNames = new HashSet<>();
			if( USE_PB_CACHE )
			{
				if( _pbcache.containsKey(pwID) ) {
					cpChildBlocks = _pbcache.get(pwID);	
				}
				else {
					cpChildBlocks = ProgramConverter.rcreateDeepCopyProgramBlocks(_childBlocks, pwID, _IDPrefix, new HashSet<String>(), fnNames, false, false); 
					_pbcache.put(pwID, cpChildBlocks);
				}
			}
			else {
				cpChildBlocks = ProgramConverter.rcreateDeepCopyProgramBlocks(_childBlocks, pwID, _IDPrefix, new HashSet<String>(), fnNames, false, false); 
			}
			
			//deep copy execution context (including prepare parfor update-in-place)
			ExecutionContext cpEc = ProgramConverter.createDeepCopyExecutionContext(ec);
			cpEc.setTID(pwID);

			// If GPU mode is enabled, gets a GPUContext from the pool of GPUContexts
			// and sets it in the ExecutionContext of the parfor
			if (DMLScript.USE_ACCELERATOR){
				cpEc.setGPUContexts(Arrays.asList(ec.getGPUContext(index)));
			}
			
			//prepare basic update-in-place variables (vars dropped on result merge)
			prepareUpdateInPlaceVariables(cpEc, pwID);
			
			//copy compiler configuration (for jmlc w/o global config)
			CompilerConfig cconf = ConfigurationManager.getCompilerConfig();
			
			//create the actual parallel worker
			ParForBody body = new ParForBody( cpChildBlocks, _resultVars, cpEc );
			pw = new LocalParWorker( pwID, queue, body, cconf, MAX_RETRYS_ON_ERROR );
			pw.setFunctionNames(fnNames);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		
		return pw;
	}
	
	/**
	 * Creates a new task partitioner according to the specified runtime parameter.
	 * 
	 * @param from ?
	 * @param to ?
	 * @param incr ?
	 * @return task partitioner
	 */
	private TaskPartitioner createTaskPartitioner( IntObject from, IntObject to, IntObject incr ) {
		return TaskPartitionerFactory.createTaskPartitioner(
			_taskPartitioner, from, to, incr, _taskSize, _numThreads, _iterPredVar);
	}
	
	/**
	 * Creates a new data partitioner according to the specified runtime parameter.
	 * 
	 * @param dpf data partition format
	 * @param dataPartitioner data partitioner
	 * @param ec execution context
	 * @return data partitioner
	 */
	private DataPartitioner createDataPartitioner(PartitionFormat dpf, PDataPartitioner dataPartitioner, ExecutionContext ec) 
	{
		DataPartitioner dp;
		
		//determine max degree of parallelism
		int numRed = OptimizerUtils.isSparkExecutionMode() ?
			SparkExecutionContext.getDefaultParallelism(false) : 1;
		
		//create data partitioner
		switch( dataPartitioner ) {
			case LOCAL:
				dp = new DataPartitionerLocal(dpf, _numThreads);
				break;
			case REMOTE_SPARK:
				dp = new DataPartitionerRemoteSpark( dpf, ec, numRed, _replicationDP, false );
				break;
			default:
				throw new DMLRuntimeException("Unknown data partitioner: '" +dataPartitioner.name()+"'.");
		}
		
		return dp;
	}

	public static ResultMerge<?> createResultMerge( PResultMerge prm,
		CacheableData<?> out, CacheableData<?>[] in, String fname, boolean accum, int numThreads, ExecutionContext ec ) 
	{
		ResultMerge<?> rm;
		
		if( out instanceof FrameObject ) {
			rm = new ResultMergeFrameLocalMemory((FrameObject)out, (FrameObject[])in, fname, accum);
		}
		else if(out instanceof MatrixObject) {
			//create result merge implementation (determine degree of parallelism 
			//only for spark to avoid unnecessary spark context creation)
			switch( prm )
			{
				case LOCAL_MEM:
					rm = new ResultMergeLocalMemory( (MatrixObject)out, (MatrixObject[])in, fname, accum );
					break;
				case LOCAL_FILE:
					rm = new ResultMergeLocalFile( (MatrixObject)out, (MatrixObject[])in, fname, accum );
					break;
				case LOCAL_AUTOMATIC:
					rm = new ResultMergeLocalAutomatic( (MatrixObject)out, (MatrixObject[])in, fname, accum );
					break;
				case REMOTE_SPARK:
					int numMap = Math.max(numThreads,
						SparkExecutionContext.getDefaultParallelism(true));
					int numRed = numMap; //equal map/reduce
					rm = new ResultMergeRemoteSpark( (MatrixObject)out,
						(MatrixObject[])in, fname, accum, ec, numMap, numRed );
					break;
				default:
					throw new DMLRuntimeException("Undefined result merge: '" + prm +"'.");
			}
		}
		else {
			throw new DMLRuntimeException("Unsupported result merge data: "+out.getClass().getSimpleName());
		}
		
		return rm;
	}
	
	/**
	 * Recompile program block hierarchy to forced CP if MR instructions or functions.
	 * Returns true if recompile was necessary and possible
	 * 
	 * @param tid thread id
	 * @return true if recompile was necessary and possible
	 */
	private boolean checkSparkAndRecompileToCP(long tid) 
	{
		//no Spark instructions, ok
		if( !OptTreeConverter.rContainsSparkInstruction(this, true) )
			return false;
		
		//no statement block, failed
		ParForStatementBlock sb = (ParForStatementBlock)getStatementBlock();
		if( sb == null ) {
			LOG.warn("Missing parfor statement block for recompile.");
			return false;
		}
		
		//try recompile Spark instructions to CP
		Recompiler.recompileProgramBlockHierarchy2Forced(
			_childBlocks, tid, new HashSet<>(), ExecType.CP);
		return true;
	}

	private void releaseForcedRecompile(long tid) {
		Recompiler.recompileProgramBlockHierarchy2Forced(
			_childBlocks, tid, new HashSet<String>(), null);
	}
	
	private void mergeLineage(ExecutionContext ec, Lineage[] lineages) {
		if( !DMLScript.LINEAGE )
			return;
		//stack lineage traces on top of each other (e.g., indexing)
		for( ResultVar var : _resultVars ) {
			LineageItem retIn = ec.getLineage().get(var._name);
			LineageItem current = lineages[0].get(var._name);
			for( int i=1; i<lineages.length; i++ ) {
				LineageItem next = lineages[i].get(var._name);
				if( next != null ) { //robustness for cond. control flow
					current = (current == null) ? next :
						LineageItemUtils.replace(next, retIn, current);
				}
			}
			ec.getLineage().set(var._name, current);
		}
	}

	private void consolidateAndCheckResults(ExecutionContext ec, final long expIters, final long expTasks,
		final long numIters, final long numTasks, LocalVariableMap[] results) {
		Timing time = new Timing(true);

		//check expected counters
		if( numTasks != expTasks || numIters !=expIters ) //consistency check
			throw new DMLRuntimeException("PARFOR: Number of executed tasks does not match the number of created tasks: tasks "+numTasks+"/"+expTasks+", iters "+numIters+"/"+expIters+".");
	
		
		//result merge
		if( checkParallelRemoteResultMerge() )
		{
			//execute result merge in parallel for all result vars
			int par = Math.min( _resultVars.size(), 
					            InfrastructureAnalyzer.getLocalParallelism() );
			if( InfrastructureAnalyzer.isLocalMode() ) {
				int parmem = (int)Math.floor(OptimizerUtils.getLocalMemBudget());
				par = Math.min(par, Math.max(parmem, 1)); //reduce k if necessary
			}
			
			try
			{
				//enqueue all result vars as tasks
				LocalTaskQueue<ResultVar> q = new LocalTaskQueue<>();
				for( ResultVar var : _resultVars ) { //foreach non-local write
					if( ec.getVariable(var._name) instanceof MatrixObject ) //robustness scalars
						q.enqueueTask(var);
				}
				q.closeInput();
				
				//run result merge workers
				ResultMergeWorker[] rmWorkers = new ResultMergeWorker[par];
				for( int i=0; i<par; i++ )
					rmWorkers[i] = new ResultMergeWorker(q, results, ec);
				for( int i=0; i<par; i++ ) //start all
					rmWorkers[i].start();
				for( int i=0; i<par; i++ ) { //wait for all
					rmWorkers[i].join();
					if( !rmWorkers[i].finishedNoError() )
						throw new DMLRuntimeException("Error occured in parallel result merge worker.");
				}
			}
			catch(Exception ex)
			{
				throw new DMLRuntimeException(ex);
			}
		}
		else
		{
			//execute result merge sequentially for all result vars
			for( ResultVar var : _resultVars ) //foreach non-local write
			{
				Data dat = ec.getVariable(var._name);
				
				if( dat instanceof MatrixObject | dat instanceof FrameObject )
				{
					CacheableData<?> out = (CacheableData<?>) dat;
					Stream<Object> tmp = Arrays.stream(results).map(vars -> vars.get(var._name));
					CacheableData<?>[] in = (dat instanceof MatrixObject) ?
						tmp.toArray(MatrixObject[]::new) : tmp.toArray(FrameObject[]::new);
					String fname = constructResultMergeFileName();
					ResultMerge<?> rm = createResultMerge(_resultMerge,
						out, in, fname, var._isAccum, _numThreads, ec);
					CacheableData<?> outNew = USE_PARALLEL_RESULT_MERGE ?
						rm.executeParallelMerge(_numThreads) :
						rm.executeSerialMerge();
					
					//cleanup existing var
					Data exdata = ec.removeVariable(var._name);
					if( exdata != null && exdata != outNew )
						ec.cleanupDataObject(exdata);
					
					//cleanup of intermediate result variables
					cleanWorkerResultVariables( ec, out, in, true );
					
					//set merged result variable
					ec.setVariable(var._name, outNew);
				}
				else if(dat instanceof ListObject) {
					ListObject oldList = (ListObject) dat;
					ListObject newList = new ListObject(oldList);
					ListObject[] in = Arrays.stream(results).map(vars -> 
						vars.get(var._name)).toArray(ListObject[]::new);
					
					//merge modified list entries into result
					for(int i=0; i<oldList.getLength(); i++) {
						Data compare = oldList.slice(i);
						for( int j=0; j<in.length; j++ ) {
							Data tmp = in[j].slice(i);
							if( compare != tmp ) {
								newList.set(i, tmp);
								break; //inner for loop
							}
						}
					}
					
					//set merged result variable
					ec.setVariable(var._name, newList);
				}
			}
		}
		
		//handle unscoped variables (vars created in parfor, but potentially used afterwards)
		ParForStatementBlock sb = (ParForStatementBlock)getStatementBlock();
		if( CREATE_UNSCOPED_RESULTVARS && sb != null && ec.getVariables() != null ) //sb might be null for nested parallelism
			createEmptyUnscopedVariables( ec.getVariables(), sb );
		
			
		if( DMLScript.STATISTICS )
			ParForStatistics.incrementMergeTime((long) time.stop());
	}
	
	/**
	 * NOTE: Currently we use a fixed rule (multiple results AND REMOTE_SPARK -> only selected by the optimizer
	 * if mode was REMOTE_SPARKJ as well). 
	 * 
	 * TODO The optimizer should explicitly decide about parallel result merge and its degree of parallelism.
	 * 
	 * @return true if ?
	 */
	private boolean checkParallelRemoteResultMerge() {
		return (USE_PARALLEL_RESULT_MERGE_REMOTE && _resultVars.size() > 1
			&& _resultMerge == PResultMerge.REMOTE_SPARK);
	}

	private void setParForProgramBlockIDs(int IDPrefix) {
		_IDPrefix = IDPrefix;
		if( _IDPrefix == -1 ) //not specified
			_ID = _pfIDSeq.getNextID(); //generated new ID
		else //remote case (further nested parfors are all in one JVM)
			_ID = IDHandler.concatIntIDsToLong(_IDPrefix, (int)_pfIDSeq.getNextID());	
	}
	
	private void setLocalParWorkerIDs() {
		if( _numThreads<=0 )
			return;
		
		//set all parworker IDs required if PExecMode.LOCAL is used
			
		_pwIDs = new long[ _numThreads ];
		
		for( int i=0; i<_numThreads; i++ ) {
			if(_IDPrefix == -1)
				_pwIDs[i] = _pwIDSeq.getNextID();
			else
				_pwIDs[i] = IDHandler.concatIntIDsToLong(_IDPrefix,(int)_pwIDSeq.getNextID());
		}
	}
	
	/**
	 * NOTE: Only required for remote parfor. Hence, there is no need to transfer DMLConfig to
	 * the remote workers (MR job) since nested remote parfor is not supported.
	 * 
	 * @return result file name
	 */
	private String constructResultFileName() {
		String scratchSpaceLoc = ConfigurationManager.getScratchSpace();
		StringBuilder sb = new StringBuilder();
		sb.append(scratchSpaceLoc);
		sb.append(Lop.FILE_SEPARATOR);
		sb.append(Lop.PROCESS_PREFIX);
		sb.append(DMLScript.getUUID());
		sb.append(PARFOR_MR_RESULT_TMP_FNAME.replaceAll("%ID%", String.valueOf(_ID)));
		
		return sb.toString();
	}

	private String constructResultMergeFileName() {
		String scratchSpaceLoc = ConfigurationManager.getScratchSpace();
		String fname = PARFOR_MR_RESULTMERGE_FNAME;
		fname = fname.replaceAll("%ID%", String.valueOf(_ID)); //replace workerID
		fname = fname.replaceAll("%VAR%", String.valueOf(_resultVarsIDSeq.getNextID()));
		
		StringBuilder sb = new StringBuilder();
		sb.append(scratchSpaceLoc);
		sb.append(Lop.FILE_SEPARATOR);
		sb.append(Lop.PROCESS_PREFIX);
		sb.append(DMLScript.getUUID());
		sb.append(fname);
		
		return sb.toString();
	}

	private String constructDataPartitionsFileName() {
		String scratchSpaceLoc = ConfigurationManager.getScratchSpace();
		String fname = PARFOR_DATAPARTITIONS_FNAME;
		fname = fname.replaceAll("%ID%", String.valueOf(_ID)); //replace workerID
		fname = fname.replaceAll("%VAR%", String.valueOf(_dpVarsIDSeq.getNextID()));
		
		StringBuilder sb = new StringBuilder();
		sb.append(scratchSpaceLoc);
		sb.append(Lop.FILE_SEPARATOR);
		sb.append(Lop.PROCESS_PREFIX);
		sb.append(DMLScript.getUUID());
		sb.append(fname);
		
		return sb.toString();
	}

	private void setMemoryBudget() {
		if( _recompileMemoryBudget > 0 ) {
			// store old budget for reset after exec
			_oldMemoryBudget = InfrastructureAnalyzer.getLocalMaxMemory();
			
			// scale budget with applied mem util factor (inverted during getMemBudget() )
			long newMaxMem = (long) (_recompileMemoryBudget / OptimizerUtils.MEM_UTIL_FACTOR);
			InfrastructureAnalyzer.setLocalMaxMemory( newMaxMem );
		}
	}
	
	private void resetMemoryBudget() {
		if( _recompileMemoryBudget > 0 )
			InfrastructureAnalyzer.setLocalMaxMemory((long)_oldMemoryBudget);
	}
	
	private void resetOptimizerFlags() {
		//reset all state that was set but is not guaranteed to be overwritten by optimizer
		_variablesDPOriginal.removeAll();
		_colocatedDPMatrix         = null;
		_replicationDP             = WRITE_REPLICATION_FACTOR;
		_replicationExport         = -1;
		_jvmReuse                  = true;
		_recompileMemoryBudget     = -1;
		_enableRuntimePiggybacking = false;
		_variablesRP               = null;
		_variablesECache           = null;
	}
	
	
	/**
	 * Helper class for parallel invocation of REMOTE_MR result merge for multiple variables.
	 */
	private class ResultMergeWorker extends Thread
	{
		private final LocalTaskQueue<ResultVar> _q;
		private final LocalVariableMap[] _refVars;
		private final ExecutionContext _ec;
		private boolean _success = false;
		
		public ResultMergeWorker( LocalTaskQueue<ResultVar> q, LocalVariableMap[] results, ExecutionContext ec )
		{
			_q = q;
			_refVars = results;
			_ec = ec;
		}
		
		public boolean finishedNoError() {
			return _success;
		}
		
		@Override
		public void run() 
		{
			try
			{
				while( true ) 
				{
					ResultVar var = _q.dequeueTask();
					if( var == LocalTaskQueue.NO_MORE_TASKS ) // task queue closed (no more tasks)
						break;
				
					CacheableData<?> out = null;
					synchronized( _ec.getVariables() ){
						out = _ec.getCacheableData(var._name);
					}
					
					Stream<Object> tmp = Arrays.stream(_refVars).map(vars -> vars.get(var._name));
					CacheableData<?>[] in = (out instanceof MatrixObject) ?
						tmp.toArray(MatrixObject[]::new) : tmp.toArray(FrameObject[]::new);
					
					String fname = constructResultMergeFileName();
				
					ResultMerge<?> rm = createResultMerge(_resultMerge,
						out, in, fname, var._isAccum, _numThreads, _ec);
					CacheableData<?> outNew;
					if( USE_PARALLEL_RESULT_MERGE )
						outNew = rm.executeParallelMerge( _numThreads );
					else
						outNew = rm.executeSerialMerge();
					
					synchronized( _ec.getVariables() ){
						_ec.getVariables().put( var._name, outNew);
					}
		
					//cleanup of intermediate result variables
					cleanWorkerResultVariables( _ec, out, in, false );
				}
				
				_success = true;
			}
			catch(Exception ex) {
				LOG.error("Error executing result merge: ", ex);
			}
		}
	}

	@Override
	public String printBlockErrorLocation(){
		return "ERROR: Runtime error in parfor program block generated from parfor statement block between lines " + _beginLine + " and " + _endLine + " -- ";
	}
}
