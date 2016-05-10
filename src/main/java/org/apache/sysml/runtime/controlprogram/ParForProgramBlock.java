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

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.log4j.Level;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.conf.CompilerConfig;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.recompile.Recompiler;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.parser.ParForStatementBlock;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.parser.VariableSet;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.DataPartitioner;
import org.apache.sysml.runtime.controlprogram.parfor.DataPartitionerLocal;
import org.apache.sysml.runtime.controlprogram.parfor.DataPartitionerRemoteMR;
import org.apache.sysml.runtime.controlprogram.parfor.DataPartitionerRemoteSpark;
import org.apache.sysml.runtime.controlprogram.parfor.LocalParWorker;
import org.apache.sysml.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysml.runtime.controlprogram.parfor.ParForBody;
import org.apache.sysml.runtime.controlprogram.parfor.ProgramConverter;
import org.apache.sysml.runtime.controlprogram.parfor.RemoteDPParForMR;
import org.apache.sysml.runtime.controlprogram.parfor.RemoteDPParForSpark;
import org.apache.sysml.runtime.controlprogram.parfor.RemoteParForJobReturn;
import org.apache.sysml.runtime.controlprogram.parfor.RemoteParForMR;
import org.apache.sysml.runtime.controlprogram.parfor.RemoteParForSpark;
import org.apache.sysml.runtime.controlprogram.parfor.ResultMerge;
import org.apache.sysml.runtime.controlprogram.parfor.ResultMergeLocalAutomatic;
import org.apache.sysml.runtime.controlprogram.parfor.ResultMergeLocalFile;
import org.apache.sysml.runtime.controlprogram.parfor.ResultMergeLocalMemory;
import org.apache.sysml.runtime.controlprogram.parfor.ResultMergeRemoteMR;
import org.apache.sysml.runtime.controlprogram.parfor.ResultMergeRemoteSpark;
import org.apache.sysml.runtime.controlprogram.parfor.Task;
import org.apache.sysml.runtime.controlprogram.parfor.TaskPartitioner;
import org.apache.sysml.runtime.controlprogram.parfor.TaskPartitionerFactoring;
import org.apache.sysml.runtime.controlprogram.parfor.TaskPartitionerFactoringCmax;
import org.apache.sysml.runtime.controlprogram.parfor.TaskPartitionerFactoringCmin;
import org.apache.sysml.runtime.controlprogram.parfor.TaskPartitionerFixedsize;
import org.apache.sysml.runtime.controlprogram.parfor.TaskPartitionerNaive;
import org.apache.sysml.runtime.controlprogram.parfor.TaskPartitionerStatic;
import org.apache.sysml.runtime.controlprogram.parfor.mqo.RuntimePiggybacking;
import org.apache.sysml.runtime.controlprogram.parfor.opt.CostEstimator;
import org.apache.sysml.runtime.controlprogram.parfor.opt.CostEstimatorHops;
import org.apache.sysml.runtime.controlprogram.parfor.opt.OptTree;
import org.apache.sysml.runtime.controlprogram.parfor.opt.OptTreeConverter;
import org.apache.sysml.runtime.controlprogram.parfor.opt.OptimizationWrapper;
import org.apache.sysml.runtime.controlprogram.parfor.opt.OptimizerRuleBased;
import org.apache.sysml.runtime.controlprogram.parfor.opt.PerfTestTool.TestMeasure;
import org.apache.sysml.runtime.controlprogram.parfor.opt.ProgramRecompiler;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.controlprogram.parfor.stat.Stat;
import org.apache.sysml.runtime.controlprogram.parfor.stat.StatisticMonitor;
import org.apache.sysml.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDHandler;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysml.runtime.instructions.cp.BooleanObject;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.DoubleObject;
import org.apache.sysml.runtime.instructions.cp.IntObject;
import org.apache.sysml.runtime.instructions.cp.StringObject;
import org.apache.sysml.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.utils.Statistics;
import org.apache.sysml.yarn.ropt.YarnClusterAnalyzer;



/**
 * The ParForProgramBlock has the same execution semantics as a ForProgamBlock but executes
 * the independent iterations in parallel. See ParForStatementBlock for the loop dependency
 * analysis. At runtime level, iterations are guaranteed to be completely independent.
 * 
 * NEW FUNCTIONALITIES (not for BI 2.0 release)
 * TODO: reduction variables (operations: +=, -=, /=, *=, min, max)
 * TODO: papply(A,1:2,FUN) language construct (compiled to ParFOR) via DML function repository => modules OK, but second-order functions required
 *
 */
public class ParForProgramBlock extends ForProgramBlock 
{	
	// execution modes
	public enum PExecMode {
		LOCAL,      //local (master) multi-core execution mode
		REMOTE_MR,	//remote (MR cluster) execution mode
		REMOTE_MR_DP,	//remote (MR cluster) execution mode, fused with data partitioning
		REMOTE_SPARK,	//remote (Spark cluster) execution mode
		REMOTE_SPARK_DP,//remote (Spark cluster) execution mode, fused with data partitioning
		UNSPECIFIED
	}

	// task partitioner
	public enum PTaskPartitioner {
		FIXED,      //fixed-sized task partitioner, uses tasksize 
		NAIVE,      //naive task partitioner (tasksize=1)
		STATIC,     //static task partitioner (numIterations/numThreads)
		FACTORING,  //factoring task partitioner  
		FACTORING_CMIN,  //constrained factoring task partitioner, uses tasksize as min constraint
		FACTORING_CMAX,  //constrained factoring task partitioner, uses tasksize as max constraint
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
		BLOCK_WISE_M_N,
		UNSPECIFIED;

		/**
		 * Note: Robust version of valueOf in order to return NONE without exception
		 * if misspelled or non-existing and for case-insensitivity.
		 * 
		 * @param s
		 * @return
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
	
	public enum PDataPartitioner {
		NONE,       // no data partitioning
		LOCAL,      // local file based partition split on master node
		REMOTE_MR,  // remote partition split using a reblock MR job 
		REMOTE_SPARK, // remote partition split using a spark job
		UNSPECIFIED, 
  	}

	public enum PResultMerge {
		LOCAL_MEM,       // in-core (in-memory) result merge (output and one input at a time)
		LOCAL_FILE,      // out-of-core result merge (file format dependent)
		LOCAL_AUTOMATIC, // decides between MEM and FILE based on the size of the output matrix 
		REMOTE_MR,       // remote MR parallel result merge
		REMOTE_SPARK,    // remote Spark parallel result merge
		UNSPECIFIED,
	}
	
	//optimizer
	public enum POptMode{
		NONE,       //no optimization, use defaults and specified parameters
		RULEBASED, //some simple rule-based rewritings (affects only parfor PB) - similar to HEURISTIC but no exec time estimates
		CONSTRAINED, //same as rule-based but with given params as constraints
		HEURISTIC, //some simple cost-based rewritings (affects only parfor PB)
		GREEDY,     //greedy cost-based optimization algorithm (potentially local optimum, affects all instructions)
		FULL_DP,    //full cost-based optimization algorithm (global optimum, affects all instructions)				
	}
		
	// internal parameters
	public static final boolean OPTIMIZE                    = true;	// run all automatic optimizations on top-level parfor
	public static final boolean USE_PB_CACHE                = false; // reuse copied program blocks whenever possible, not there can be issues related to recompile
	public static       boolean USE_RANGE_TASKS_IF_USEFUL   = true;   	// use range tasks whenever size>3, false, otherwise wrong split order in remote 
	public static final boolean USE_STREAMING_TASK_CREATION = true;  	// start working while still creating tasks, prevents blocking due to too small task queue
	public static final boolean ALLOW_NESTED_PARALLELISM	= true;    // if not, transparently change parfor to for on program conversions (local,remote)
	public static       boolean ALLOW_REUSE_MR_JVMS         = true;    // potential benefits: less setup costs per task, NOTE> cannot be used MR4490 in Hadoop 1.0.3, still not fixed in 1.1.1
	public static       boolean ALLOW_REUSE_MR_PAR_WORKER   = ALLOW_REUSE_MR_JVMS; //potential benefits: less initialization, reuse in-memory objects and result consolidation!
	public static final boolean USE_PARALLEL_RESULT_MERGE   = false;    // if result merge is run in parallel or serial 
	public static final boolean USE_PARALLEL_RESULT_MERGE_REMOTE = true; // if remote result merge should be run in parallel for multiple result vars
	public static final boolean ALLOW_DATA_COLOCATION       = true;
	public static final boolean CREATE_UNSCOPED_RESULTVARS  = true;
	public static       boolean ALLOW_REUSE_PARTITION_VARS  = true; //reuse partition input matrices, applied only if read-only in surrounding loops
	public static final int     WRITE_REPLICATION_FACTOR    = 1;
	public static final int     MAX_RETRYS_ON_ERROR         = 1;
	public static final boolean FORCE_CP_ON_REMOTE_MR       = true; // compile body to CP if exec type forced to MR
	public static final boolean LIVEVAR_AWARE_EXPORT        = true; //export only read variables according to live variable analysis
 	public static final boolean LIVEVAR_AWARE_CLEANUP       = true; //cleanup pinned variables according to live variable analysis
	public static final boolean RESET_RECOMPILATION_FLAGs   = true;
 	
 	public static final String PARFOR_FNAME_PREFIX          = "/parfor/"; 
	public static final String PARFOR_MR_TASKS_TMP_FNAME    = PARFOR_FNAME_PREFIX + "%ID%_MR_taskfile"; 
	public static final String PARFOR_MR_RESULT_TMP_FNAME   = PARFOR_FNAME_PREFIX + "%ID%_MR_results"; 
	public static final String PARFOR_MR_RESULTMERGE_FNAME  = PARFOR_FNAME_PREFIX + "%ID%_resultmerge%VAR%"; 
	public static final String PARFOR_DATAPARTITIONS_FNAME  = PARFOR_FNAME_PREFIX + "%ID%_datapartitions%VAR%"; 
	
	public static final String PARFOR_COUNTER_GROUP_NAME    = "SystemML ParFOR Counters";
	
	// static ID generator sequences
	private static IDSequence   _pfIDSeq        = null;
	private static IDSequence   _pwIDSeq        = null;
	
	// runtime parameters
	protected HashMap<String,String> _params    = null;
	protected int              _numThreads      = -1;
	protected PTaskPartitioner _taskPartitioner = null; 
	protected long             _taskSize        = -1;
	protected PDataPartitioner _dataPartitioner = null;
	protected PResultMerge     _resultMerge     = null;
	protected PExecMode        _execMode        = null;
	protected POptMode         _optMode         = null;
	protected boolean          _monitor         = false;
	protected Level            _optLogLevel     = null;
	
	
	//specifics used for optimization
	protected long             _numIterations   = -1; 
	protected String[]         _iterablePredicateVarsOriginal = null;
	
	//specifics used for data partitioning
	protected LocalVariableMap _variablesDPOriginal = null;
	protected LocalVariableMap _variablesDPReuse    = null;
	protected String           _colocatedDPMatrix   = null;
	protected boolean          _tSparseCol          = false;
	protected int              _replicationDP       = WRITE_REPLICATION_FACTOR;
	protected int              _replicationExport   = -1;
	//specifics used for result partitioning
	protected boolean          _jvmReuse            = true;
	//specifics used for recompilation 
	protected double           _oldMemoryBudget = -1;
	protected double           _recompileMemoryBudget = -1;
	//specifics for caching
	protected boolean          _enableCPCaching     = true;
	protected boolean          _enableRuntimePiggybacking = false;
	//specifics for spark 
	protected Collection<String> _variablesRP = null;
	protected Collection<String> _variablesECache = null;
	
	// program block meta data
	protected long                _ID           = -1;
	protected int                 _IDPrefix     = -1;
	protected ArrayList<String>  _resultVars      = null;
	protected IDSequence         _resultVarsIDSeq = null;
	protected IDSequence         _dpVarsIDSeq     = null;
	protected boolean            _monitorReport   = false;
	protected boolean            _hasFunctions    = true;
	
	// local parworker data
	protected long[] 		   	                    _pwIDs   = null;
	protected HashMap<Long,ArrayList<ProgramBlock>> _pbcache = null;
	
	
	static
	{
		//init static ID sequence generators
		_pfIDSeq = new IDSequence();
		_pwIDSeq = new IDSequence();
	}
	
	public ParForProgramBlock(Program prog, String[] iterPredVars, HashMap<String,String> params) 
		throws DMLRuntimeException 
	{
		this( -1, prog, iterPredVars, params);
	}
	
	/**
	 * ParForProgramBlock constructor. It reads the specified parameter settings, where defaults for non-specified parameters
	 * have been set in ParForStatementBlock.validate(). Furthermore, it generates the IDs for the ParWorkers.
	 * 
	 * @param prog
	 * @param iterPred
	 * @throws DMLRuntimeException 
	 */
	public ParForProgramBlock(int ID, Program prog, String[] iterPredVars, HashMap<String,String> params) 
		throws DMLRuntimeException  
	{
		super(prog, iterPredVars);

		//init internal flags according to DML config
		initInternalConfigurations(ConfigurationManager.getDMLConfig());
		
		//ID generation and setting 
		setParForProgramBlockIDs( ID );
		_resultVarsIDSeq = new IDSequence();
		_dpVarsIDSeq = new IDSequence();
		
		//parse and use internal parameters (already set to default if not specified)
		_params = params;
		try
		{
			_numThreads      = Integer.parseInt( _params.get(ParForStatementBlock.PAR) );
			_taskPartitioner = PTaskPartitioner.valueOf( _params.get(ParForStatementBlock.TASK_PARTITIONER).toUpperCase() );
			_taskSize        = Integer.parseInt( _params.get(ParForStatementBlock.TASK_SIZE) );
			_dataPartitioner = PDataPartitioner.valueOf( _params.get(ParForStatementBlock.DATA_PARTITIONER).toUpperCase() );
			_resultMerge     = PResultMerge.valueOf( _params.get(ParForStatementBlock.RESULT_MERGE).toUpperCase() );
			_execMode        = PExecMode.valueOf( _params.get(ParForStatementBlock.EXEC_MODE).toUpperCase() );
			_optMode         = POptMode.valueOf( _params.get(ParForStatementBlock.OPT_MODE).toUpperCase());		
			_optLogLevel     = Level.toLevel( _params.get(ParForStatementBlock.OPT_LOG));
			_monitor         = (Integer.parseInt(_params.get(ParForStatementBlock.PROFILE) ) == 1);
		}
		catch(Exception ex)
		{
			//runtime exception in order to keep signature of program block
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
			_pbcache = new HashMap<Long, ArrayList<ProgramBlock>>();
		
		//created profiling report after parfor exec
		_monitorReport = _monitor;
		
		//materialized meta data (reused for all invocations)
		_hasFunctions = ProgramRecompiler.containsAtLeastOneFunction(this);
		
		LOG.trace("PARFOR: ParForProgramBlock created with mode = "+_execMode+", optmode = "+_optMode+", numThreads = "+_numThreads);
	}
	
	public long getID()
	{
		return _ID;
	}
	
	public PExecMode getExecMode()
	{
		return _execMode;
	}
	
	public HashMap<String,String> getParForParams()
	{
		return _params;
	}

	public ArrayList<String> getResultVariables()
	{
		return _resultVars;
	}
	
	public void setResultVariables(ArrayList<String> resultVars)
	{
		_resultVars = resultVars;
	}
	
	public void disableOptimization()
	{
		_optMode = POptMode.NONE;
	}
	
	public POptMode getOptimizationMode()
	{
		return _optMode;
	}
	
	public int getDegreeOfParallelism()
	{
		return _numThreads;
	}
	
	public void setDegreeOfParallelism(int k)
	{
		_numThreads = k;
		_params.put(ParForStatementBlock.PAR, String.valueOf(_numThreads)); //kept up-to-date for copies
		setLocalParWorkerIDs();
	}

	public void setCPCaching(boolean flag)
	{
		_enableCPCaching = flag;
	}
	
	public void setRuntimePiggybacking(boolean flag)
	{
		_enableRuntimePiggybacking = flag;
	}
	
	public void setExecMode( PExecMode mode )
	{
		_execMode = mode;
		_params.put(ParForStatementBlock.EXEC_MODE, String.valueOf(_execMode)); //kept up-to-date for copies
	}
	
	public void setTaskPartitioner( PTaskPartitioner partitioner )
	{
		_taskPartitioner = partitioner;
		_params.put(ParForStatementBlock.TASK_PARTITIONER, String.valueOf(_taskPartitioner)); //kept up-to-date for copies
	}
	
	public void setTaskSize( long tasksize )
	{
		_taskSize = tasksize;
		_params.put(ParForStatementBlock.TASK_SIZE, String.valueOf(_taskSize)); //kept up-to-date for copies
	}
	
	public void setDataPartitioner(PDataPartitioner partitioner) 
	{
		_dataPartitioner = partitioner;
		_params.put(ParForStatementBlock.DATA_PARTITIONER, String.valueOf(_dataPartitioner)); //kept up-to-date for copies
	}
	
	public void enableColocatedPartitionedMatrix( String varname )
	{
		//only called from optimizer
		_colocatedDPMatrix = varname;
	}
	
	public void setTransposeSparseColumnVector( boolean flag )
	{
		_tSparseCol = flag;
	}
	
	public void setPartitionReplicationFactor( int rep )
	{
		//only called from optimizer
		_replicationDP = rep;
	}
	
	public void setExportReplicationFactor( int rep )
	{
		//only called from optimizer
		_replicationExport = rep;
	}
	
	public void disableJVMReuse() 
	{
		//only called from optimizer
		_jvmReuse = false;
	}
	
	public void disableMonitorReport()
	{
		_monitorReport = false;
	}
	
	public void setResultMerge(PResultMerge merge) 
	{
		_resultMerge = merge;
		_params.put(ParForStatementBlock.RESULT_MERGE, String.valueOf(_resultMerge)); //kept up-to-date for copies
	}
	
	public void setRecompileMemoryBudget( double localMem )
	{
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
	
	public long getNumIterations()
	{
		return _numIterations;
	}
	
	public boolean hasFunctions() {
		return _hasFunctions;
	}

	public static void initInternalConfigurations( DMLConfig conf )
	{
		ALLOW_REUSE_MR_JVMS = conf.getBooleanValue(DMLConfig.JVM_REUSE);
		ALLOW_REUSE_MR_PAR_WORKER = ALLOW_REUSE_MR_JVMS;
	}
	
	@Override	
	public void execute(ExecutionContext ec)
		throws DMLRuntimeException
	{	
		ParForStatementBlock sb = (ParForStatementBlock)getStatementBlock();
		
		// add the iterable predicate variable to the variable set
		String iterVarName = _iterablePredicateVars[0];

		// evaluate from, to, incr only once (assumption: known at for entry)
		IntObject from = executePredicateInstructions( 1, _fromInstructions, ec );
		IntObject to   = executePredicateInstructions( 2, _toInstructions, ec );
		IntObject incr = (_incrementInstructions == null || _incrementInstructions.isEmpty()) && _iterablePredicateVars[3]==null ? 
				new IntObject((from.getLongValue()<=to.getLongValue()) ? 1 : -1) :
				executePredicateInstructions( 3, _incrementInstructions, ec );
		
		if ( incr.getLongValue() == 0 ) //would produce infinite loop
			throw new DMLRuntimeException(this.printBlockErrorLocation() + "Expression for increment of variable '" + iterVarName + "' must evaluate to a non-zero value.");
		
		//early exit on num iterations = zero
		if( computeNumIterations(from, to, incr) <= 0 )
			return; //avoid unnecessary optimization/initialization
		
		///////
		//OPTIMIZATION of ParFOR body (incl all child parfor PBs)
		///////
		if( _optMode != POptMode.NONE )
		{
			updateIterablePredicateVars( iterVarName, from, to, incr );
			OptimizationWrapper.setLogLevel(_optLogLevel); //set optimizer log level
			OptimizationWrapper.optimize( _optMode, sb, this, ec, _monitor ); //core optimize
			
			//take changed iterable predicate into account
			iterVarName = _iterablePredicateVars[0];
			from = executePredicateInstructions( 1, _fromInstructions, ec );
			to   = executePredicateInstructions( 2, _toInstructions, ec );
			incr = executePredicateInstructions( 3, _incrementInstructions, ec );
		}
		
		///////
		//DATA PARTITIONING of read-only parent variables of type (matrix,unpartitioned)
		///////
		Timing time = _monitor ? new Timing(true) : null;
		
		//partitioning on demand (note: for fused data partitioning and execute the optimizer set 
		//the data partitioner to NONE in order to prevent any side effects)
		handleDataPartitioning( ec ); 
	
		//repartitioning of variables for spark cpmm/zipmm in order prevent unnecessary shuffle
		handleSparkRepartitioning( ec );
		
		//eager rdd caching of variables for spark in order prevent read/write contention
		handleSparkEagerCaching( ec );
		
		if( _monitor ) 
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_INIT_DATA_T, time.stop());
			
		// initialize iter var to form value
		IntObject iterVar = new IntObject(iterVarName, from.getLongValue() );
		
		///////
		//begin PARALLEL EXECUTION of (PAR)FOR body
		///////
		LOG.trace("EXECUTE PARFOR ID = "+_ID+" with mode = "+_execMode+", numThreads = "+_numThreads+", taskpartitioner = "+_taskPartitioner);
		
		if( _monitor )
		{
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_NUMTHREADS,      _numThreads);
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_TASKSIZE,        _taskSize);
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_TASKPARTITIONER, _taskPartitioner.ordinal());
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_DATAPARTITIONER, _dataPartitioner.ordinal());
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_EXECMODE,        _execMode.ordinal());
		}
		
		//preserve shared input/result variables of cleanup
		ArrayList<String> varList = ec.getVarList();
		HashMap<String, Boolean> varState = ec.pinVariables(varList);
		
		try 
		{		
			switch( _execMode )
			{
				case LOCAL: //create parworkers as local threads
					executeLocalParFor(ec, iterVar, from, to, incr);
					break;
					
				case REMOTE_MR: // create parworkers as MR tasks (one job per parfor)
					executeRemoteMRParFor(ec, iterVar, from, to, incr);
					break;
				
				case REMOTE_MR_DP: // create parworkers as MR tasks (one job per parfor)
					executeRemoteMRParForDP(ec, iterVar, from, to, incr);
					break;
				
				case REMOTE_SPARK: // create parworkers as Spark tasks (one job per parfor)
					executeRemoteSparkParFor(ec, iterVar, from, to, incr);
					break;
				
				case REMOTE_SPARK_DP: // create parworkers as Spark tasks (one job per parfor)
					executeRemoteSparkParForDP(ec, iterVar, from, to, incr);
					break;
				
				default:
					throw new DMLRuntimeException("Undefined execution mode: '"+_execMode+"'.");
			}	
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("PARFOR: Failed to execute loop in parallel.",ex);
		}
		
		//reset state of shared input/result variables 
		ec.unpinVariables(varList, varState);
		
		//cleanup unpinned shared variables
		cleanupSharedVariables(ec, varState);
		
		//set iteration var to TO value (+ increment) for FOR equivalence
		iterVar = new IntObject( iterVarName, to.getLongValue() ); //consistent with for
		ec.setVariable(iterVarName, iterVar);
		
		//ensure that subsequent program blocks never see partitioned data (invalid plans!)
		//we can replace those variables, because partitioning only applied for read-only matrices
		for( String var : _variablesDPOriginal.keySet() )
		{
			//cleanup partitioned matrix (if not reused)
			if( !_variablesDPReuse.keySet().contains(var) )
				VariableCPInstruction.processRemoveVariableInstruction(ec, var); 
			//reset to original matrix
			MatrixObject mo = (MatrixObject) _variablesDPOriginal.get( var );
			ec.setVariable(var, mo); 
		}
		
		///////
		//end PARALLEL EXECUTION of (PAR)FOR body
		///////
	
		//print profiling report (only if top-level parfor because otherwise in parallel context)
		if( _monitorReport )
		    LOG.info("\n"+StatisticMonitor.createReport());
		
		//reset flags/modifications made by optimizer
		//TODO reset of hop parallelism constraint (e.g., ba+*)
		for( String dpvar : _variablesDPOriginal.keySet() ) //release forced exectypes
		    ProgramRecompiler.rFindAndRecompileIndexingHOP(sb, this, dpvar, ec, false);
		 //release forced exectypes for fused dp/exec
		if( _execMode == PExecMode.REMOTE_MR_DP || _execMode == PExecMode.REMOTE_SPARK_DP )
			ProgramRecompiler.rFindAndRecompileIndexingHOP(sb, this, _colocatedDPMatrix, ec, false); 
		resetIterablePredicateVars();
		resetOptimizerFlags(); //after release, deletes dp_varnames
		
		//execute exit instructions (usually empty)
		executeInstructions(_exitInstructions, ec);			
	}


	/**
	 * Executes the parfor locally, i.e., the parfor is realized with numThreads local threads that drive execution.
	 * This execution mode allows for arbitrary nested local parallelism and nested invocations of MR jobs. See
	 * below for details of the realization.
	 * 
	 * @param ec
	 * @param itervar
	 * @param from
	 * @param to
	 * @param incr
	 * @throws DMLRuntimeException
	 * @throws InterruptedException 
	 */
	private void executeLocalParFor( ExecutionContext ec, IntObject itervar, IntObject from, IntObject to, IntObject incr ) 
		throws DMLRuntimeException, InterruptedException
	{
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
		
		//enable runtime piggybacking if required
		if( _enableRuntimePiggybacking )
			RuntimePiggybacking.start( _numThreads ); //default piggybacking worker
		
		try
		{
			// Step 1) init parallel workers, task queue and threads
			LocalTaskQueue<Task> queue = new LocalTaskQueue<Task>();
			Thread[] threads         = new Thread[_numThreads];
			LocalParWorker[] workers = new LocalParWorker[_numThreads];
			for( int i=0; i<_numThreads; i++ )
			{
				//create parallel workers as (lazy) deep copies
				workers[i] = createParallelWorker( _pwIDs[i], queue, ec ); 
				threads[i] = new Thread( workers[i] );
				threads[i].setPriority(Thread.MAX_PRIORITY); 
			}
			
			// start threads (from now on waiting for tasks)
			for( Thread thread : threads )
				thread.start();
			
			//maintain statistics
			long tinit = (long) time.stop();
			if( DMLScript.STATISTICS )
				Statistics.incrementParForInitTime(tinit);
			if( _monitor ) 
				StatisticMonitor.putPFStat(_ID, Stat.PARFOR_INIT_PARWRK_T, tinit);
			
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
			if( _monitor )
				StatisticMonitor.putPFStat(_ID, Stat.PARFOR_INIT_TASKS_T, time.stop());
			
			// Step 3) join all threads (wait for finished work)
			for( Thread thread : threads )
				thread.join();
			
			if( _monitor ) 
				StatisticMonitor.putPFStat(_ID, Stat.PARFOR_WAIT_EXEC_T, time.stop());
				
				
			// Step 4) collecting results from each parallel worker
			//obtain results
			LocalVariableMap [] localVariables = new LocalVariableMap [_numThreads]; 
			for( int i=0; i<_numThreads; i++ )
			{
				localVariables[i] = workers[i].getVariables();
				numExecutedTasks += workers[i].getExecutedTasks();
				numExecutedIterations += workers[i].getExecutedIterations();			
			}
			//consolidate results into global symbol table
			consolidateAndCheckResults( ec, numIterations, numCreatedTasks, numExecutedIterations, numExecutedTasks, 
					                    localVariables );
			
			// Step 5) cleanup local parworkers (e.g., remove created functions)
			for( int i=0; i<_numThreads; i++ )
			{
				Collection<String> fnNames = workers[i].getFunctionNames();
				if( fnNames!=null && !fnNames.isEmpty() )
					for( String fn : fnNames ) {
						String[] parts = DMLProgram.splitFunctionKey(fn);
						_prog.removeFunctionProgramBlock(parts[0], parts[1]);
					}
			}
		}
		finally 
		{
			//remove thread-local memory budget (reset to original budget)
			//(in finally to prevent error side effects for multiple scripts in one jvm)
			resetMemoryBudget();
		
			//disable runtime piggybacking
			if( _enableRuntimePiggybacking )
				RuntimePiggybacking.stop();
			
			if( _monitor )  {
				StatisticMonitor.putPFStat(_ID, Stat.PARFOR_WAIT_RESULTS_T, time.stop());
				StatisticMonitor.putPFStat(_ID, Stat.PARFOR_NUMTASKS, numExecutedTasks);
				StatisticMonitor.putPFStat(_ID, Stat.PARFOR_NUMITERS, numExecutedIterations);
			}
		}
	}	
	
	/**
	 * 
	 * @param ec
	 * @param itervar
	 * @param from
	 * @param to
	 * @param incr
	 * @throws DMLRuntimeException
	 * @throws IOException 
	 */
	private void executeRemoteMRParFor( ExecutionContext ec, IntObject itervar, IntObject from, IntObject to, IntObject incr ) 
		throws DMLRuntimeException, IOException
	{
		/* Step 0) check and recompile MR inst
		 * Step 1) serialize child PB and inst
		 * Step 2) create tasks
		 *         serialize tasks
		 * Step 3) submit MR Jobs and wait for results                        
		 * Step 4) collect results from each parallel worker
		 */
		
		Timing time = ( _monitor ? new Timing(true) : null );
		
		// Step 0) check and compile to CP (if forced remote parfor)
		boolean flagForced = false;
		if( FORCE_CP_ON_REMOTE_MR && (_optMode == POptMode.NONE || (_optMode == POptMode.CONSTRAINED && _execMode==PExecMode.REMOTE_MR)) )
		{
			//tid = 0  because replaced in remote parworker
			flagForced = checkMRAndRecompileToCP(0);
		}
			
		// Step 1) init parallel workers (serialize PBs)
		// NOTES: each mapper changes filenames with regard to his ID as we submit a single job,
		//        cannot reuse serialized string, since variables are serialized as well.
		ParForBody body = new ParForBody( _childBlocks, _resultVars, ec );
		String program = ProgramConverter.serializeParForBody( body );
		
		if( _monitor ) 
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_INIT_PARWRK_T, time.stop());
		
		// Step 2) create tasks 
		TaskPartitioner partitioner = createTaskPartitioner(from, to, incr);
		String taskFile = constructTaskFileName();
		String resultFile = constructResultFileName();
		
		long numIterations = partitioner.getNumIterations();
		int maxDigits = (int)Math.log10(to.getLongValue()) + 1;
		long numCreatedTasks = -1;
		if( USE_STREAMING_TASK_CREATION )
		{
			LocalTaskQueue<Task> queue = new LocalTaskQueue<Task>();

			//put tasks into queue and start writing to taskFile
			numCreatedTasks = partitioner.createTasks(queue);
			taskFile        = writeTasksToFile( taskFile, queue, maxDigits );				
		}
		else
		{
			//sequentially create tasks and write to disk
			List<Task> tasks = partitioner.createTasks();
			numCreatedTasks  = tasks.size();
		    taskFile         = writeTasksToFile( taskFile, tasks, maxDigits );				
		}
				
		if( _monitor )
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_INIT_TASKS_T, time.stop());
		
		//write matrices to HDFS 
		exportMatricesToHDFS(ec);
				
		// Step 3) submit MR job (wait for finished work)
		MatrixObject colocatedDPMatrixObj = (_colocatedDPMatrix!=null)? (MatrixObject)ec.getVariable(_colocatedDPMatrix) : null;
		RemoteParForJobReturn ret = RemoteParForMR.runJob(_ID, program, taskFile, resultFile, colocatedDPMatrixObj, _enableCPCaching,
				                                          _numThreads, WRITE_REPLICATION_FACTOR, MAX_RETRYS_ON_ERROR, getMinMemory(ec),
				                                          (ALLOW_REUSE_MR_JVMS & _jvmReuse) );
		
		if( _monitor ) 
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_WAIT_EXEC_T, time.stop());
			
			
		// Step 4) collecting results from each parallel worker
		int numExecutedTasks = ret.getNumExecutedTasks();
		int numExecutedIterations = ret.getNumExecutedIterations();
		
		//consolidate results into global symbol table
		consolidateAndCheckResults( ec, numIterations, numCreatedTasks, numExecutedIterations , numExecutedTasks, 
				                    ret.getVariables() );
		if( flagForced ) //see step 0
			releaseForcedRecompile(0);
		
		if( _monitor ) 
		{
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_WAIT_RESULTS_T, time.stop());
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_NUMTASKS, numExecutedTasks);
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_NUMITERS, numExecutedIterations);
		}			
	}	
	
	/**
	 * 
	 * @param ec
	 * @param itervar
	 * @param from
	 * @param to
	 * @param incr
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
	private void executeRemoteMRParForDP( ExecutionContext ec, IntObject itervar, IntObject from, IntObject to, IntObject incr ) 
		throws DMLRuntimeException, IOException
	{
		/* Step 0) check and recompile MR inst
		 * Step 1) serialize child PB and inst
		 * Step 2) create tasks
		 *         serialize tasks
		 * Step 3) submit MR Jobs and wait for results                        
		 * Step 4) collect results from each parallel worker
		 */
		
		Timing time = ( _monitor ? new Timing(true) : null );
		
		// Step 0) check and compile to CP (if forced remote parfor)
		boolean flagForced = checkMRAndRecompileToCP(0);
		
		// Step 1) prepare partitioned input matrix (needs to happen before serializing the progam)
		ParForStatementBlock sb = (ParForStatementBlock) getStatementBlock();
		MatrixObject inputMatrix = (MatrixObject)ec.getVariable(_colocatedDPMatrix );
		PDataPartitionFormat inputDPF = sb.determineDataPartitionFormat( _colocatedDPMatrix );
		inputMatrix.setPartitioned(inputDPF, 1); //mark matrix var as partitioned (for reducers) 
		
		// Step 2) init parallel workers (serialize PBs)
		// NOTES: each mapper changes filenames with regard to his ID as we submit a single job,
		//        cannot reuse serialized string, since variables are serialized as well.
		ParForBody body = new ParForBody( _childBlocks, _resultVars, ec );
		String program = ProgramConverter.serializeParForBody( body );
		
		if( _monitor ) 
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_INIT_PARWRK_T, time.stop());
		
		// Step 3) create tasks 
		TaskPartitioner partitioner = createTaskPartitioner(from, to, incr);
		String resultFile = constructResultFileName();
		long numIterations = partitioner.getNumIterations();
		long numCreatedTasks = numIterations;//partitioner.createTasks().size();
						
		if( _monitor )
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_INIT_TASKS_T, time.stop());
		
		//write matrices to HDFS 
		exportMatricesToHDFS(ec);
				
		// Step 4) submit MR job (wait for finished work)
		OutputInfo inputOI = ((inputMatrix.getSparsity()<0.1 && inputDPF==PDataPartitionFormat.COLUMN_WISE)||
				              (inputMatrix.getSparsity()<0.001 && inputDPF==PDataPartitionFormat.ROW_WISE))? 
				             OutputInfo.BinaryCellOutputInfo : OutputInfo.BinaryBlockOutputInfo;
		RemoteParForJobReturn ret = RemoteDPParForMR.runJob(_ID, itervar.getName(), _colocatedDPMatrix, program, resultFile, 
				inputMatrix, inputDPF, inputOI, _tSparseCol, _enableCPCaching, _numThreads, _replicationDP, MAX_RETRYS_ON_ERROR );
		
		if( _monitor ) 
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_WAIT_EXEC_T, time.stop());
			
		// Step 5) collecting results from each parallel worker
		int numExecutedTasks = ret.getNumExecutedTasks();
		int numExecutedIterations = ret.getNumExecutedIterations();
		
		//consolidate results into global symbol table
		consolidateAndCheckResults( ec, numIterations, numCreatedTasks, numExecutedIterations, numExecutedTasks, 
				                    ret.getVariables() );
		
		if( flagForced ) //see step 0
			releaseForcedRecompile(0);
		inputMatrix.unsetPartitioned();
		
		if( _monitor ) 
		{
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_WAIT_RESULTS_T, time.stop());
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_NUMTASKS, numExecutedTasks);
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_NUMITERS, numExecutedIterations);
		}			
	}
	
	/**
	 * 
	 * @param ec
	 * @param itervar
	 * @param from
	 * @param to
	 * @param incr
	 * @throws DMLRuntimeException 
	 */
	private void executeRemoteSparkParFor(ExecutionContext ec, IntObject itervar, IntObject from, IntObject to, IntObject incr) 
		throws DMLRuntimeException
	{
		Timing time = ( _monitor ? new Timing(true) : null );
		
		// Step 0) check and compile to CP (if forced remote parfor)
		boolean flagForced = false;
		if( FORCE_CP_ON_REMOTE_MR && (_optMode == POptMode.NONE || (_optMode == POptMode.CONSTRAINED && _execMode==PExecMode.REMOTE_SPARK)) )
		{
			//tid = 0  because replaced in remote parworker
			flagForced = checkMRAndRecompileToCP(0); 
		}
			
		// Step 1) init parallel workers (serialize PBs)
		// NOTES: each mapper changes filenames with regard to his ID as we submit a single job,
		//        cannot reuse serialized string, since variables are serialized as well.
		ParForBody body = new ParForBody( _childBlocks, _resultVars, ec );
		String program = ProgramConverter.serializeParForBody( body );
		
		if( _monitor ) 
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_INIT_PARWRK_T, time.stop());
		
		// Step 2) create tasks 
		TaskPartitioner partitioner = createTaskPartitioner(from, to, incr);
		long numIterations = partitioner.getNumIterations();
		
		//sequentially create tasks as input to parfor job
		List<Task> tasks = partitioner.createTasks();
		long numCreatedTasks = tasks.size();
				
		if( _monitor )
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_INIT_TASKS_T, time.stop());
		
		//write matrices to HDFS 
		exportMatricesToHDFS(ec);
				
		// Step 3) submit Spark parfor job (no lazy evaluation, since collect on result)
		//MatrixObject colocatedDPMatrixObj = (_colocatedDPMatrix!=null)? (MatrixObject)ec.getVariable(_colocatedDPMatrix) : null;
		RemoteParForJobReturn ret = RemoteParForSpark.runJob(_ID, program, tasks, ec, _enableCPCaching, _numThreads);
		
		if( _monitor ) 
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_WAIT_EXEC_T, time.stop());
			
			
		// Step 4) collecting results from each parallel worker
		int numExecutedTasks = ret.getNumExecutedTasks();
		int numExecutedIterations = ret.getNumExecutedIterations();
		
		//consolidate results into global symbol table
		consolidateAndCheckResults( ec, numIterations, numCreatedTasks, numExecutedIterations , numExecutedTasks, 
				                    ret.getVariables() );
		if( flagForced ) //see step 0
			releaseForcedRecompile(0);
		
		if( _monitor ) 
		{
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_WAIT_RESULTS_T, time.stop());
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_NUMTASKS, numExecutedTasks);
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_NUMITERS, numExecutedIterations);
		}			
	}
	
	private void executeRemoteSparkParForDP( ExecutionContext ec, IntObject itervar, IntObject from, IntObject to, IntObject incr ) 
		throws DMLRuntimeException, IOException
	{
		Timing time = ( _monitor ? new Timing(true) : null );
		
		// Step 0) check and compile to CP (if forced remote parfor)
		boolean flagForced = checkMRAndRecompileToCP(0);
		
		// Step 1) prepare partitioned input matrix (needs to happen before serializing the progam)
		ParForStatementBlock sb = (ParForStatementBlock) getStatementBlock();
		MatrixObject inputMatrix = (MatrixObject)ec.getVariable(_colocatedDPMatrix );
		PDataPartitionFormat inputDPF = sb.determineDataPartitionFormat( _colocatedDPMatrix );
		inputMatrix.setPartitioned(inputDPF, 1); //mark matrix var as partitioned (for reducers) 
		
		// Step 2) init parallel workers (serialize PBs)
		// NOTES: each mapper changes filenames with regard to his ID as we submit a single job,
		//        cannot reuse serialized string, since variables are serialized as well.
		ParForBody body = new ParForBody( _childBlocks, _resultVars, ec );
		String program = ProgramConverter.serializeParForBody( body );
		
		if( _monitor ) 
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_INIT_PARWRK_T, time.stop());
		
		// Step 3) create tasks 
		TaskPartitioner partitioner = createTaskPartitioner(from, to, incr);
		String resultFile = constructResultFileName();
		long numIterations = partitioner.getNumIterations();
		long numCreatedTasks = numIterations;//partitioner.createTasks().size();
						
		if( _monitor )
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_INIT_TASKS_T, time.stop());
		
		//write matrices to HDFS 
		exportMatricesToHDFS(ec);
				
		// Step 4) submit MR job (wait for finished work)
		OutputInfo inputOI = ((inputMatrix.getSparsity()<0.1 && inputDPF==PDataPartitionFormat.COLUMN_WISE)||
				              (inputMatrix.getSparsity()<0.001 && inputDPF==PDataPartitionFormat.ROW_WISE))? 
				             OutputInfo.BinaryCellOutputInfo : OutputInfo.BinaryBlockOutputInfo;
		RemoteParForJobReturn ret = RemoteDPParForSpark.runJob(_ID, itervar.getName(), _colocatedDPMatrix, program, resultFile, 
				inputMatrix, ec, inputDPF, inputOI, _tSparseCol, _enableCPCaching, _numThreads );
		
		if( _monitor ) 
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_WAIT_EXEC_T, time.stop());
			
		// Step 5) collecting results from each parallel worker
		int numExecutedTasks = ret.getNumExecutedTasks();
		int numExecutedIterations = ret.getNumExecutedIterations();
		
		//consolidate results into global symbol table
		consolidateAndCheckResults( ec, numIterations, numCreatedTasks, numExecutedIterations, numExecutedTasks, 
				                    ret.getVariables() );
		
		if( flagForced ) //see step 0
			releaseForcedRecompile(0);
		inputMatrix.unsetPartitioned();
		
		if( _monitor ) 
		{
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_WAIT_RESULTS_T, time.stop());
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_NUMTASKS, numExecutedTasks);
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_NUMITERS, numExecutedIterations);
		}			
	}
	
	/**
	 * 
	 * @param ec
	 * @throws DMLRuntimeException 
	 */
	private void handleDataPartitioning( ExecutionContext ec ) 
		throws DMLRuntimeException
	{
		if( _dataPartitioner != PDataPartitioner.NONE )
		{			
			ParForStatementBlock sb = (ParForStatementBlock) getStatementBlock();
			if( sb == null )
				throw new DMLRuntimeException("ParFor statement block required for reasoning about data partitioning.");
			
			ArrayList<String> vars = sb.getReadOnlyParentVars();
			for( String var : vars )
			{
				Data dat = ec.getVariable(var);
				//skip non-existing input matrices (which are due to unknown sizes marked for
				//partitioning but typically related branches are never executed)				
				if( dat != null && dat instanceof MatrixObject )
				{
					MatrixObject moVar = (MatrixObject) dat; //unpartitioned input
					
					PDataPartitionFormat dpf = sb.determineDataPartitionFormat( var );
					//dpf = (_optMode != POptMode.NONE) ? OptimizerRuleBased.decideBlockWisePartitioning(moVar, dpf) : dpf;
					LOG.trace("PARFOR ID = "+_ID+", Partitioning read-only input variable "+var+" (format="+dpf+", mode="+_dataPartitioner+")");
					
					if( dpf != PDataPartitionFormat.NONE )
					{
						Timing ltime = new Timing(true);
						
						//input data partitioning (reuse if possible)
						Data dpdatNew = _variablesDPReuse.get(var);
						if( dpdatNew == null ) //no reuse opportunity
						{
							DataPartitioner dp = createDataPartitioner( dpf, _dataPartitioner, ec );
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
						if(    ALLOW_REUSE_PARTITION_VARS 
							&& ProgramRecompiler.isApplicableForReuseVariable(sb.getDMLProg(), sb, var) ) 
						{
							_variablesDPReuse.put(var, dpdatNew);
						}
						
						LOG.trace("Partitioning and recompilation done in "+ltime.stop()+"ms");
					}
				}
			}
		}
	}
	
	/**
	 * 
	 * @param ec
	 * @throws DMLRuntimeException
	 */
	private void handleSparkRepartitioning( ExecutionContext ec ) 
		throws DMLRuntimeException
	{
		if( OptimizerUtils.isSparkExecutionMode() &&
			_variablesRP != null && !_variablesRP.isEmpty() ) {
			SparkExecutionContext sec = (SparkExecutionContext) ec;
			
			for( String var : _variablesRP )
				sec.repartitionAndCacheMatrixObject(var);
		}
	}
	
	/**
	 * 
	 * @param ec
	 * @throws DMLRuntimeException
	 */
	private void handleSparkEagerCaching( ExecutionContext ec ) 
		throws DMLRuntimeException
	{
		if( OptimizerUtils.isSparkExecutionMode() &&
			_variablesECache != null && !_variablesECache.isEmpty() ) {
			SparkExecutionContext sec = (SparkExecutionContext) ec;
			
			for( String var : _variablesECache )
				sec.cacheMatrixObject(var);
		}
	}
	
	/**
	 * Cleanup result variables of parallel workers after result merge.
	 * @param in 
	 * @param out 
	 * @throws DMLRuntimeException 
	 */
	private void cleanWorkerResultVariables(ExecutionContext ec, MatrixObject out, MatrixObject[] in) 
		throws DMLRuntimeException
	{
		for( MatrixObject tmp : in )
		{
			//check for empty inputs (no iterations executed)
			if( tmp != null && tmp != out )
				ec.cleanupMatrixObject(tmp);
		}
	}
	
	/**
	 * Create empty matrix objects and scalars for all unscoped vars 
	 * (created within the parfor).
	 * 
	 * NOTE: parfor gives no guarantees on the values of those objects - hence
	 * we return -1 for sclars and empty matrix objects.
	 * 
	 * @param out
	 * @param sb
	 * @throws DMLRuntimeException 
	 */
	private void createEmptyUnscopedVariables( LocalVariableMap out, StatementBlock sb ) 
		throws DMLRuntimeException
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
						switch( valuetype )
						{
							case BOOLEAN: dataObj = new BooleanObject(var,false); break;
							case INT:     dataObj = new IntObject(var,-1);        break;
							case DOUBLE:  dataObj = new DoubleObject(var,-1d);    break;
							case STRING:  dataObj = new StringObject(var,"-1");   break;
							default:
								throw new DMLRuntimeException("Value type not supported: "+valuetype);
						}
						break;
					case MATRIX:
						//currently we do not create any unscoped matrix object outputs
						//because metadata (e.g., outputinfo) not known at this place.
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
	
	/**
	 * 
	 * @throws CacheException
	 */
	private void exportMatricesToHDFS( ExecutionContext ec ) 
		throws CacheException 
	{
		ParForStatementBlock sb = (ParForStatementBlock)getStatementBlock();
		
		if( LIVEVAR_AWARE_EXPORT && sb != null)
		{
			//optimization to prevent unnecessary export of matrices
			//export only variables that are read in the body
			VariableSet varsRead = sb.variablesRead();
			for (String key : ec.getVariables().keySet() ) 
			{
				Data d = ec.getVariable(key);
				if (    d.getDataType() == DataType.MATRIX
					 && varsRead.containsVariable(key)  )
				{
					MatrixObject mo = (MatrixObject)d;
					mo.exportData( _replicationExport );
				}
			}
		}
		else
		{
			//export all matrices in symbol table
			for (String key : ec.getVariables().keySet() ) 
			{
				Data d = ec.getVariable(key);
				if ( d.getDataType() == DataType.MATRIX )
				{
					MatrixObject mo = (MatrixObject)d;
					mo.exportData( _replicationExport );
				}
			}
		}
	}
	
	/**
	 * 
	 * @param ec
	 * @param varState
	 * @throws DMLRuntimeException
	 */
	private void cleanupSharedVariables( ExecutionContext ec, HashMap<String,Boolean> varState ) 
		throws DMLRuntimeException 
	{
		//TODO needs as precondition a systematic treatment of persistent read information.
		/*
		if( LIVEVAR_AWARE_CLEANUP && _sb != null)
		{
			//cleanup shared variables after they are unpinned
			VariableSet liveout = _sb.liveOut();
			for( Entry<String, Boolean> var : varState.entrySet() ) 
			{
				String varname = var.getKey();
				boolean unpinned = var.getValue();
				String fprefix = ConfigurationManager.getConfig().getTextValue("scratch") 
						         + Lop.FILE_SEPARATOR + Lop.PROCESS_PREFIX + DMLScript.getUUID();
				
				//delete unpinned vars if not in liveout (similar like rmvar) and not persistent input
				if( unpinned && !liveout.containsVariable(varname) )
					      
				{
					VariableCPInstruction.processRemoveVariableInstruction(ec,varname);
				}
			}
		}
		*/
	}
	
	/**
	 * Creates a new or partially recycled instance of a parallel worker. Therefore the symbol table, and child
	 * program blocks are deep copied. Note that entries of the symbol table are not deep copied because they are replaced 
	 * anyway on the next write. In case of recycling the deep copies of program blocks are recycled from previous 
	 * executions of this parfor.
	 * 
	 * 
	 * @param pwID
	 * @param queue
	 * @param ec
	 * @return
	 * @throws InstantiationException
	 * @throws IllegalAccessException
	 * @throws DMLRuntimeException
	 * @throws CloneNotSupportedException
	 */
	private LocalParWorker createParallelWorker(long pwID, LocalTaskQueue<Task> queue, ExecutionContext ec) 
		throws DMLRuntimeException
	{
		LocalParWorker pw = null; 
		
		try
		{
			//create deep copies of required elements child blocks
			ArrayList<ProgramBlock> cpChildBlocks = null;	
			HashSet<String> fnNames = new HashSet<String>();
			if( USE_PB_CACHE )
			{
				if( _pbcache.containsKey(pwID) )
				{
					cpChildBlocks = _pbcache.get(pwID);	
				}
				else
				{
					cpChildBlocks = ProgramConverter.rcreateDeepCopyProgramBlocks(_childBlocks, pwID, _IDPrefix, new HashSet<String>(), fnNames, false, false); 
					_pbcache.put(pwID, cpChildBlocks);
				}
			}
			else
			{
				cpChildBlocks = ProgramConverter.rcreateDeepCopyProgramBlocks(_childBlocks, pwID, _IDPrefix, new HashSet<String>(), fnNames, false, false); 
			}             
			
			//deep copy execution context
			ExecutionContext cpEc = ProgramConverter.createDeepCopyExecutionContext(ec);
			
			//copy compiler configuration (for jmlc w/o global config)
			CompilerConfig cconf = ConfigurationManager.getCompilerConfig();
			
			//create the actual parallel worker
			ParForBody body = new ParForBody( cpChildBlocks, _resultVars, cpEc );
			pw = new LocalParWorker( pwID, queue, body, cconf, MAX_RETRYS_ON_ERROR, _monitor );
			pw.setFunctionNames(fnNames);
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		
		return pw;
	}
	
	/**
	 * Creates a new task partitioner according to the specified runtime parameter.
	 * 
	 * @param from
	 * @param to
	 * @param incr
	 * @return
	 * @throws DMLRuntimeException
	 */
	private TaskPartitioner createTaskPartitioner( IntObject from, IntObject to, IntObject incr ) 
		throws DMLRuntimeException
	{
		TaskPartitioner tp = null;
		
		switch( _taskPartitioner )
		{
			case FIXED:
				tp = new TaskPartitionerFixedsize( _taskSize, _iterablePredicateVars[0],
	                    					   from, to, incr );
				break;
			case NAIVE:
				tp = new TaskPartitionerNaive( _taskSize, _iterablePredicateVars[0],
                        					   from, to, incr );
				break;
			case STATIC:
				tp = new TaskPartitionerStatic( _taskSize, _numThreads, _iterablePredicateVars[0],
                        					   from, to, incr );
				break;
			case FACTORING:
				tp = new TaskPartitionerFactoring( _taskSize,_numThreads, _iterablePredicateVars[0],
							                       from, to, incr );
				break;
			case FACTORING_CMIN:
				//for constrained factoring the tasksize is used as the minimum constraint
				tp = new TaskPartitionerFactoringCmin( _taskSize,_numThreads, _taskSize, _iterablePredicateVars[0],
							                       from, to, incr );
				break;

			case FACTORING_CMAX:
				//for constrained factoring the tasksize is used as the minimum constraint
				tp = new TaskPartitionerFactoringCmax( _taskSize,_numThreads, _taskSize, _iterablePredicateVars[0],
							                       from, to, incr );
				break;	
			default:
				throw new DMLRuntimeException("Undefined task partitioner: '"+_taskPartitioner+"'.");
		}
		
		return tp;
	}
	
	/**
	 * Creates a new data partitioner according to the specified runtime parameter.
	 * 
	 * @param dpf
	 * @param dataPartitioner
	 * @param ec 
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private DataPartitioner createDataPartitioner(PDataPartitionFormat dpf, PDataPartitioner dataPartitioner, ExecutionContext ec) 
		throws DMLRuntimeException 
	{
		DataPartitioner dp = null;
		
		//determine max degree of parallelism
		int numReducers = ConfigurationManager.getNumReducers();
		int maxNumRed = InfrastructureAnalyzer.getRemoteParallelReduceTasks();
		//correction max number of reducers on yarn clusters
		if( InfrastructureAnalyzer.isYarnEnabled() )
			maxNumRed = (int)Math.max( maxNumRed, YarnClusterAnalyzer.getNumCores()/2 );				
		int numRed = Math.min(numReducers,maxNumRed);
		
		//create data partitioner
		switch( dataPartitioner )
		{
			case LOCAL:
				dp = new DataPartitionerLocal(dpf, -1, _numThreads);
				break;
			case REMOTE_MR:
				dp = new DataPartitionerRemoteMR( dpf, -1, _ID, numRed,
						                          _replicationDP, 
						                          MAX_RETRYS_ON_ERROR, 
						                          ALLOW_REUSE_MR_JVMS, false );
				break;
			case REMOTE_SPARK:
				dp = new DataPartitionerRemoteSpark( dpf, -1, ec, numRed, false );
				break;	
			default:
				throw new DMLRuntimeException("Undefined data partitioner: '" +dataPartitioner.toString()+"'.");
		}
		
		return dp;
	}
	
	/**
	 * 
	 * @param prm
	 * @param out
	 * @param in
	 * @param fname
	 * @return
	 * @throws DMLRuntimeException
	 */
	private ResultMerge createResultMerge( PResultMerge prm, MatrixObject out, MatrixObject[] in, String fname, ExecutionContext ec ) 
		throws DMLRuntimeException 
	{
		ResultMerge rm = null;
		
		//determine degree of parallelism
		int numReducers = ConfigurationManager.getNumReducers();
		int maxMap = InfrastructureAnalyzer.getRemoteParallelMapTasks();
		int maxRed = InfrastructureAnalyzer.getRemoteParallelReduceTasks();
		//correction max number of reducers on yarn clusters
		if( InfrastructureAnalyzer.isYarnEnabled() ) {					
			maxMap = (int)Math.max( maxMap, YarnClusterAnalyzer.getNumCores() );	
			maxRed = (int)Math.max( maxRed, YarnClusterAnalyzer.getNumCores()/2 );	
		}
		int numMap = Math.max(_numThreads, maxMap);
		int numRed = Math.min(numReducers, maxRed);
		
		//create result merge implementation		
		switch( prm )
		{
			case LOCAL_MEM:
				rm = new ResultMergeLocalMemory( out, in, fname );
				break;
			case LOCAL_FILE:
				rm = new ResultMergeLocalFile( out, in, fname );
				break;
			case LOCAL_AUTOMATIC:
				rm = new ResultMergeLocalAutomatic( out, in, fname );
				break;
			case REMOTE_MR:
				rm = new ResultMergeRemoteMR( out, in, fname, _ID, numMap, numRed,
					                          WRITE_REPLICATION_FACTOR, 
					                          MAX_RETRYS_ON_ERROR, 
					                          ALLOW_REUSE_MR_JVMS );
				break;
			case REMOTE_SPARK:
				rm = new ResultMergeRemoteSpark( out, in, fname, ec, numMap, numRed );
				break;
				
			default:
				throw new DMLRuntimeException("Undefined result merge: '" +prm.toString()+"'.");
		}
		
		return rm;
	}
	
	/**
	 * Recompile program block hierarchy to forced CP if MR instructions or functions.
	 * Returns true if recompile was necessary and possible
	 * 
	 * @param tid
	 * @return
	 * @throws DMLRuntimeException
	 */
	private boolean checkMRAndRecompileToCP(long tid) 
		throws DMLRuntimeException
	{
		//no MR instructions, ok
		if( !OptTreeConverter.rContainsMRJobInstruction(this, true) )
			return false;
		
		//no statement block, failed
		ParForStatementBlock sb = (ParForStatementBlock)getStatementBlock();
		if( sb == null ) {
			LOG.warn("Missing parfor statement block for recompile.");
			return false;
		}
		
		//try recompile MR instructions to CP
		HashSet<String> fnStack = new HashSet<String>();
		Recompiler.recompileProgramBlockHierarchy2Forced(_childBlocks, tid, fnStack, ExecType.CP);
		return true;
	}
	
	/**
	 * 
	 * @param tid
	 * @throws DMLRuntimeException
	 */
	private void releaseForcedRecompile(long tid) 
		throws DMLRuntimeException
	{
		HashSet<String> fnStack = new HashSet<String>();
		Recompiler.recompileProgramBlockHierarchy2Forced(_childBlocks, tid, fnStack, null);
	}
	
	
	/**
	 * 
	 * @param fname
	 * @param tasks
	 * @return
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
	private String writeTasksToFile(String fname, List<Task> tasks, int maxDigits)
		throws DMLRuntimeException, IOException
	{
		BufferedWriter br = null;
		try
		{
			Path path = new Path(fname);
			FileSystem fs = FileSystem.get(ConfigurationManager.getCachedJobConf());
			br = new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));
	        
			boolean flagFirst = true; //workaround for keeping gen order
			for( Task t : tasks )
			{
				br.write( createTaskFileLine( t, maxDigits, flagFirst ) );
				if( flagFirst )
					flagFirst = false;
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Error writing tasks to taskfile "+fname, ex);
		}
		finally
		{
			if( br !=null )
				br.close();
		}
		
		return fname;
	}
	
	/**
	 * 
	 * @param fname
	 * @param queue
	 * @return
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
	private String writeTasksToFile(String fname, LocalTaskQueue<Task> queue, int maxDigits)
		throws DMLRuntimeException, IOException
	{
		BufferedWriter br = null;
		try
		{
			Path path = new Path( fname );
			FileSystem fs = FileSystem.get(ConfigurationManager.getCachedJobConf());
			br = new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));
	        
			Task t = null;
			boolean flagFirst = true; //workaround for keeping gen order
			while( (t = queue.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS )
			{
				br.write( createTaskFileLine( t, maxDigits, flagFirst ) );
				if( flagFirst )
					flagFirst = false;
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Error writing tasks to taskfile "+fname, ex);
		}
		finally
		{
			if( br !=null )
				br.close();
		}
		
		return fname;
	}
	
	private String createTaskFileLine( Task t, int maxDigits, boolean flagFirst ) 
	{
		//always pad to max digits in order to preserve task order	
		String ret = t.toCompactString(maxDigits) + (flagFirst?" ":"") + "\n";
		return ret;
	}
	
	/**
	 * 
	 * @param expIters
	 * @param expTasks
	 * @param numIters
	 * @param numTasks
	 * @param results
	 * @throws DMLRuntimeException
	 */
	private void consolidateAndCheckResults(ExecutionContext ec, long expIters, long expTasks, long numIters, long numTasks, LocalVariableMap [] results) 
		throws DMLRuntimeException
	{
		Timing time = new Timing(true);
		
		//result merge
		if( checkParallelRemoteResultMerge() )
		{
			//execute result merge in parallel for all result vars
			int par = Math.min( _resultVars.size(), 
					            InfrastructureAnalyzer.getLocalParallelism() );
			if( InfrastructureAnalyzer.isLocalMode() ) {
				int parmem = (int)Math.floor(OptimizerUtils.getLocalMemBudget() / 
						InfrastructureAnalyzer.getRemoteMaxMemorySortBuffer());
				par = Math.min(par, Math.max(parmem, 1)); //reduce k if necessary
			}
			
			try
			{
				//enqueue all result vars as tasks
				LocalTaskQueue<String> q = new LocalTaskQueue<String>();
				for( String var : _resultVars ) //foreach non-local write
					if( ec.getVariable(var) instanceof MatrixObject ) //robustness scalars
						q.enqueueTask(var);
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
			for( String var : _resultVars ) //foreach non-local write
			{			
				Data dat = ec.getVariable(var);
				if( dat instanceof MatrixObject ) //robustness scalars
				{
					MatrixObject out = (MatrixObject) dat;
					MatrixObject[] in = new MatrixObject[ results.length ];
					for( int i=0; i< results.length; i++ )
						in[i] = (MatrixObject) results[i].get( var ); 			
					String fname = constructResultMergeFileName();
					ResultMerge rm = createResultMerge(_resultMerge, out, in, fname, ec);
					MatrixObject outNew = null;
					if( USE_PARALLEL_RESULT_MERGE )
						outNew = rm.executeParallelMerge( _numThreads );
					else
						outNew = rm.executeSerialMerge(); 		
					
					//cleanup existing var
					Data exdata = ec.removeVariable(var);
					if( exdata != null && exdata != outNew && exdata instanceof MatrixObject )
						ec.cleanupMatrixObject((MatrixObject)exdata);
							
					//cleanup of intermediate result variables
					cleanWorkerResultVariables( ec, out, in );
					
					//set merged result variable
					ec.setVariable(var, outNew);
				}
			}
		}
		
		//handle unscoped variables (vars created in parfor, but potentially used afterwards)
		ParForStatementBlock sb = (ParForStatementBlock)getStatementBlock();
		if( CREATE_UNSCOPED_RESULTVARS && sb != null && ec.getVariables() != null ) //sb might be null for nested parallelism
			createEmptyUnscopedVariables( ec.getVariables(), sb );
		
		//check expected counters
		if( numTasks != expTasks || numIters !=expIters ) //consistency check
			throw new DMLRuntimeException("PARFOR: Number of executed tasks does not match the number of created tasks: tasks "+numTasks+"/"+expTasks+", iters "+numIters+"/"+expIters+".");
	
		if( DMLScript.STATISTICS )
			Statistics.incrementParForMergeTime((long) time.stop());
	}
	
	/**
	 * NOTE: Currently we use a fixed rule (multiple results AND REMOTE_MR -> only selected by the optimizer
	 * if mode was REMOTE_MR as well). 
	 * 
	 * TODO The optimizer should explicitly decide about parallel result merge and its degree of parallelism.
	 * 
	 * @return
	 */
	private boolean checkParallelRemoteResultMerge()
	{
		return (USE_PARALLEL_RESULT_MERGE_REMOTE 
			    && _resultVars.size() > 1
			    && ( _resultMerge == PResultMerge.REMOTE_MR
			       ||_resultMerge == PResultMerge.REMOTE_SPARK) );
	}
	
	/**
	 * 
	 * @param IDPrefix
	 */
	private void setParForProgramBlockIDs(int IDPrefix)
	{
		_IDPrefix = IDPrefix;
		if( _IDPrefix == -1 ) //not specified
			_ID = _pfIDSeq.getNextID(); //generated new ID
		else //remote case (further nested parfors are all in one JVM)
		    _ID = IDHandler.concatIntIDsToLong(_IDPrefix, (int)_pfIDSeq.getNextID());	
	}
	
	/**
	 * TODO rework id handling in order to enable worker reuse
	 * 
	 */
	private void setLocalParWorkerIDs()
	{
		if( _numThreads<=0 )
			return;
		
		//set all parworker IDs required if PExecMode.LOCAL is used
			
		_pwIDs = new long[ _numThreads ];
		
		for( int i=0; i<_numThreads; i++ )
		{
			if(_IDPrefix == -1)
				_pwIDs[i] = _pwIDSeq.getNextID();
			else
				_pwIDs[i] = IDHandler.concatIntIDsToLong(_IDPrefix,(int)_pwIDSeq.getNextID());
			
			if( _monitor ) 
				StatisticMonitor.putPfPwMapping(_ID, _pwIDs[i]);
		}
	}

	/**
	 * 
	 * @param from
	 * @param to
	 * @param incr
	 */
	private long computeNumIterations( IntObject from, IntObject to, IntObject incr )
	{
		return (long)Math.ceil(((double)(to.getLongValue() - from.getLongValue() + 1)) / incr.getLongValue()); 
	}
	
	/**
	 * 
	 * @param iterVarName
	 * @param from
	 * @param to
	 * @param incr
	 */
	private void updateIterablePredicateVars(String iterVarName, IntObject from, IntObject to, IntObject incr) 
	{
		_numIterations = computeNumIterations(from, to, incr); 
		
		//keep original iterable predicate
		_iterablePredicateVarsOriginal = new String[4];
		System.arraycopy(_iterablePredicateVars, 0, _iterablePredicateVarsOriginal, 0, 4);
		
		_iterablePredicateVars[0] = iterVarName;
		_iterablePredicateVars[1] = from.getStringValue();
		_iterablePredicateVars[2] = to.getStringValue();
		_iterablePredicateVars[3] = incr.getStringValue();
	}
	
	/**
	 * 
	 */
	private void resetIterablePredicateVars()
	{
		//reset of modified for optimization (opt!=NONE)
		if( _iterablePredicateVarsOriginal!=null ) 
			System.arraycopy(_iterablePredicateVarsOriginal, 0, _iterablePredicateVars, 0, 4);
	}
	
	/**
	 * NOTE: Only required for remote parfor. Hence, there is no need to transfer DMLConfig to
	 * the remote workers (MR job) since nested remote parfor is not supported.
 	 * 
	 * @return
	 */
	private String constructTaskFileName()
	{
		String scratchSpaceLoc = ConfigurationManager.getScratchSpace();
	
		StringBuilder sb = new StringBuilder();
		sb.append(scratchSpaceLoc);
		sb.append(Lop.FILE_SEPARATOR);
		sb.append(Lop.PROCESS_PREFIX);
		sb.append(DMLScript.getUUID());
		sb.append(PARFOR_MR_TASKS_TMP_FNAME.replaceAll("%ID%", String.valueOf(_ID)));
		
		return sb.toString();   
	}
	
	/**
	 * NOTE: Only required for remote parfor. Hence, there is no need to transfer DMLConfig to
	 * the remote workers (MR job) since nested remote parfor is not supported.
	 * 
	 * @return
	 */
	private String constructResultFileName()
	{
		String scratchSpaceLoc = ConfigurationManager.getScratchSpace();
		
		StringBuilder sb = new StringBuilder();
		sb.append(scratchSpaceLoc);
		sb.append(Lop.FILE_SEPARATOR);
		sb.append(Lop.PROCESS_PREFIX);
		sb.append(DMLScript.getUUID());
		sb.append(PARFOR_MR_RESULT_TMP_FNAME.replaceAll("%ID%", String.valueOf(_ID)));
		
		return sb.toString();   
	}

	/**
	 * 
	 * @return
	 */
	private String constructResultMergeFileName()
	{
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
	
	/**
	 * 
	 * @return
	 */
	private String constructDataPartitionsFileName()
	{
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

	/**
	 * 
	 * @return
	 */
	private long getMinMemory(ExecutionContext ec)
	{
		long ret = -1;
		
		//if forced remote exec and single node
		if(    DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE 
			&& _execMode == PExecMode.REMOTE_MR
			&& _optMode == POptMode.NONE      )
		{
			try 
			{
				ParForStatementBlock sb = (ParForStatementBlock)getStatementBlock();
				OptTree tree = OptTreeConverter.createAbstractOptTree(-1, -1, sb, this, new HashSet<String>(), ec);
				CostEstimator est = new CostEstimatorHops( OptTreeConverter.getAbstractPlanMapping() );
				double mem = est.getEstimate(TestMeasure.MEMORY_USAGE, tree.getRoot());
				
				ret = (long) (mem * ( 1d/OptimizerUtils.MEM_UTIL_FACTOR  )); 
			} 
			catch(Exception e) 
			{
				LOG.error("Failed to analyze minmum memory requirements.", e);
			} 
		}
		
		return ret;
	}

	private void setMemoryBudget()
	{
		if( _recompileMemoryBudget > 0 )
		{
			// store old budget for reset after exec
			_oldMemoryBudget = (double)InfrastructureAnalyzer.getLocalMaxMemory();
			
			// scale budget with applied mem util factor (inverted during getMemBudget() )
			long newMaxMem = (long) (_recompileMemoryBudget / OptimizerUtils.MEM_UTIL_FACTOR);
			InfrastructureAnalyzer.setLocalMaxMemory( newMaxMem );
		}
	}
	
	private void resetMemoryBudget()
	{
		if( _recompileMemoryBudget > 0 )
		{
			InfrastructureAnalyzer.setLocalMaxMemory((long)_oldMemoryBudget);
		}
	}
	
	private void resetOptimizerFlags()
	{
		//reset all state that was set but is not guaranteed to be overwritten by optimizer
		_variablesDPOriginal.removeAll();
		_iterablePredicateVarsOriginal = null;
		_colocatedDPMatrix     = null;
		_replicationDP         = WRITE_REPLICATION_FACTOR;
		_replicationExport     = -1;
		_jvmReuse              = true;
		_recompileMemoryBudget = -1;
		_enableRuntimePiggybacking = false;
		_variablesRP           = null;
		_variablesECache       = null;
	}
	
	
	/**
	 * Helper class for parallel invocation of REMOTE_MR result merge for multiple variables.
	 */
	private class ResultMergeWorker extends Thread
	{
		private LocalTaskQueue<String> _q = null;
		private LocalVariableMap[] _refVars = null;
		private ExecutionContext _ec = null;
		private boolean _success = false;
		
		public ResultMergeWorker( LocalTaskQueue<String> q, LocalVariableMap[] results, ExecutionContext ec )
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
					String varname = _q.dequeueTask();
					if( varname == LocalTaskQueue.NO_MORE_TASKS ) // task queue closed (no more tasks)
						break;
				
					MatrixObject out = null;
					synchronized( _ec.getVariables() ){
						out = (MatrixObject) _ec.getVariable(varname);
					}
					
					MatrixObject[] in = new MatrixObject[ _refVars.length ];
					for( int i=0; i< _refVars.length; i++ )
						in[i] = (MatrixObject) _refVars[i].get( varname ); 			
					String fname = constructResultMergeFileName();
				
					ResultMerge rm = createResultMerge(_resultMerge, out, in, fname, _ec);
					MatrixObject outNew = null;
					if( USE_PARALLEL_RESULT_MERGE )
						outNew = rm.executeParallelMerge( _numThreads );
					else
						outNew = rm.executeSerialMerge(); 	
					
					synchronized( _ec.getVariables() ){
						_ec.getVariables().put( varname, outNew);
					}
		
					//cleanup of intermediate result variables
					cleanWorkerResultVariables( _ec, out, in );
				}
				
				_success = true;
			}
			catch(Exception ex)
			{
				LOG.error("Error executing result merge: ", ex);
			}
		}
	}
	

	public String printBlockErrorLocation(){
		return "ERROR: Runtime error in parfor program block generated from parfor statement block between lines " + _beginLine + " and " + _endLine + " -- ";
	}
	
}