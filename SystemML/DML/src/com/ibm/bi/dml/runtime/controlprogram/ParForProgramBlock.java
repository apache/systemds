/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.compile.Recompiler;
import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.ParForStatementBlock;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.VariableSet;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheException;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.DataPartitioner;
import com.ibm.bi.dml.runtime.controlprogram.parfor.DataPartitionerLocal;
import com.ibm.bi.dml.runtime.controlprogram.parfor.DataPartitionerRemoteMR;
import com.ibm.bi.dml.runtime.controlprogram.parfor.LocalParWorker;
import com.ibm.bi.dml.runtime.controlprogram.parfor.LocalTaskQueue;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ParForBody;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.RemoteParForJobReturn;
import com.ibm.bi.dml.runtime.controlprogram.parfor.RemoteParForMR;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ResultMerge;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ResultMergeLocalAutomatic;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ResultMergeLocalFile;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ResultMergeLocalMemory;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ResultMergeRemoteMR;
import com.ibm.bi.dml.runtime.controlprogram.parfor.Task;
import com.ibm.bi.dml.runtime.controlprogram.parfor.TaskPartitioner;
import com.ibm.bi.dml.runtime.controlprogram.parfor.TaskPartitionerFactoring;
import com.ibm.bi.dml.runtime.controlprogram.parfor.TaskPartitionerFactoringCmax;
import com.ibm.bi.dml.runtime.controlprogram.parfor.TaskPartitionerFactoringCmin;
import com.ibm.bi.dml.runtime.controlprogram.parfor.TaskPartitionerFixedsize;
import com.ibm.bi.dml.runtime.controlprogram.parfor.TaskPartitionerNaive;
import com.ibm.bi.dml.runtime.controlprogram.parfor.TaskPartitionerStatic;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.CostEstimator;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.CostEstimatorHops;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptTree;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptTreeConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptimizationWrapper;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.ProgramRecompiler;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.TestMeasure;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Stat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.StatisticMonitor;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Timing;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDHandler;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.StringObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.VariableCPInstruction;



/**
 * The ParForProgramBlock has the same execution semantics as a ForProgamBlock but executes
 * the independent iterations in parallel. See ParForStatementBlock for the loop dependency
 * analysis. At runtime level, iterations are guaranteed to be completely independent.
 * 
 * TODO currently it seams that there is a general issue with JVM reuse in Hadoop 1.0.3
 * (Error Task log directory for task attempt_201209251949_2586_m_000014_0 does not exist. May be cleaned up by Task Tracker, if older logs.)      
 * 
 * NEW FUNCTIONALITIES (not for BI 2.0 release)
 * TODO: reduction variables (operations: +=, -=, /=, *=, min, max)
 * TODO: papply(A,1:2,FUN) language construct (compiled to ParFOR) via DML function repository => modules OK, but second-order functions required
 *
 */
public class ParForProgramBlock extends ForProgramBlock 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	// execution modes
	public enum PExecMode{
		LOCAL,      //local (master) multi-core execution mode
		REMOTE_MR,	//remote (MR cluster) execution mode	
		UNSPECIFIED
	}

	// task partitioner
	public enum PTaskPartitioner{
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
		UNSPECIFIED
  	}

	public enum PResultMerge {
		LOCAL_MEM,  // in-core (in-memory) result merge (output and one input at a time)
		LOCAL_FILE, // out-of-core result merge (file format dependent)
		LOCAL_AUTOMATIC, // decides between MEM and FILE based on the size of the output matrix 
		REMOTE_MR, // remote parallel result merge
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
	public static final boolean MONITOR                     = false;	// collect internal statistics
	public static final boolean OPTIMIZE                    = true;	// run all automatic optimizations on top-level parfor
	public static final boolean USE_PB_CACHE                = true;  	// reuse copied program blocks whenever possible
	public static       boolean USE_RANGE_TASKS_IF_USEFUL   = true;   	// use range tasks whenever size>3, false, otherwise wrong split order in remote 
	public static final boolean USE_STREAMING_TASK_CREATION = true;  	// start working while still creating tasks, prevents blocking due to too small task queue
	public static final boolean ALLOW_NESTED_PARALLELISM	= true;    // if not, transparently change parfor to for on program conversions (local,remote)
	public static final boolean ALLOW_REUSE_MR_JVMS         = false;     // potential benefits: less setup costs per task, NOTE> cannot be used MR4490 in Hadoop 1.0.3, still not fixed in 1.1.1
	public static final boolean ALLOW_REUSE_MR_PAR_WORKER   = ALLOW_REUSE_MR_JVMS; //potential benefits: less initialization, reuse in-memory objects and result consolidation!
	public static final boolean USE_FLEX_SCHEDULER_CONF     = false;
	public static final boolean USE_PARALLEL_RESULT_MERGE   = false;    // if result merge is run in parallel or serial 
	public static final boolean USE_PARALLEL_RESULT_MERGE_REMOTE = true; // if remote result merge should be run in parallel for multiple result vars
	public static final boolean ALLOW_DATA_COLOCATION       = true;
	public static final boolean CREATE_UNSCOPED_RESULTVARS  = true;
	public static final boolean ALLOW_UNSCOPED_PARTITIONING = false;
	public static final int     WRITE_REPLICATION_FACTOR    = 1;
	public static final int     MAX_RETRYS_ON_ERROR         = 1;
	public static final boolean FORCE_CP_ON_REMOTE_MR       = true; // compile body to CP if exec type forced to MR
	public static final boolean LIVEVAR_AWARE_EXPORT        = true; //export only read variables according to live variable analysis
 	public static final boolean LIVEVAR_AWARE_CLEANUP       = true; //cleanup pinned variables according to live variable analysis
	
	
	public static final String PARFOR_MR_TASKS_TMP_FNAME    = "/parfor/%ID%_MR_taskfile.dat"; 
	public static final String PARFOR_MR_RESULT_TMP_FNAME   = "/parfor/%ID%_MR_results.dat"; 
	public static final String PARFOR_MR_RESULTMERGE_FNAME   = "/parfor/%ID%_resultmerge%VAR%.dat"; 
	
	// static ID generator sequences
	private static IDSequence   _pfIDSeq        = null;
	private static IDSequence   _pwIDSeq        = null;
	
	// runtime parameters
	protected HashMap<String,String> _params    = null;
	protected int              _numThreads      = -1;
	protected PTaskPartitioner _taskPartitioner = null; 
	protected int              _taskSize        = -1;
	protected PDataPartitioner _dataPartitioner = null;
	protected PResultMerge     _resultMerge     = null;
	protected PExecMode        _execMode        = null;
	protected POptMode         _optMode         = null;
	
	//specifics used for optimization
	protected ParForStatementBlock _sb          = null;
	protected int              _numIterations   = -1; 
	
	//specifics used for data partitioning
	protected LocalVariableMap _variablesDPOriginal = null;
	protected String           _colocatedDPMatrix   = null;
	protected int              _replicationDP       = WRITE_REPLICATION_FACTOR;
	protected int              _replicationExport   = -1;
	//specifics used for result partitioning
	protected boolean          _jvmReuse            = true;
	//specifics used for recompilation 
	protected double           _oldMemoryBudget = -1;
	protected double           _recompileMemoryBudget = -1;
	
	// program block meta data
	protected long                _ID           = -1;
	protected int                 _IDPrefix     = -1;
	protected ArrayList<String>  _resultVars      = null;
	protected IDSequence         _resultVarsIDSeq = null;
	
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
	
		//ID generation and setting 
		setParForProgramBlockIDs( ID );
		_resultVarsIDSeq = new IDSequence();
		
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
		}
		catch(Exception ex)
		{
			//runtime exception in order to keep signature of program block
			throw new RuntimeException("Error parsing specified ParFOR parameters.",ex);
		}
			
		//reset the internal opt mode if optimization globally disabled.
		if( !OPTIMIZE )
			_optMode = POptMode.NONE;
			
		if( !ALLOW_UNSCOPED_PARTITIONING )
			_variablesDPOriginal = new LocalVariableMap();
		
		
		//create IDs for all parworkers
		if( _execMode == PExecMode.LOCAL )
			setLocalParWorkerIDs();
	
		//initialize program block cache if necessary
		if( USE_PB_CACHE ) 
			_pbcache = new HashMap<Long, ArrayList<ProgramBlock>>();
		
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
	
	public void setTaskSize( int tasksize )
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
		//only enabled though optimizer
		_colocatedDPMatrix = varname;
	}
	
	public void setPartitionReplicationFactor( int rep )
	{
		_replicationDP = rep;
	}
	
	public void setExportReplicationFactor( int rep )
	{
		_replicationExport = rep;
	}
	
	public void disableJVMReuse() 
	{
		_jvmReuse = false;
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
	
	public int getNumIterations()
	{
		return _numIterations;
	}

	public ParForStatementBlock getStatementBlock( )
	{
		return _sb;
	}
	
	public void setStatementBlock( ParForStatementBlock sb )
	{
		_sb = sb;
	}
	
	@Override	
	public void execute(ExecutionContext ec)
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{	
		// add the iterable predicate variable to the variable set
		String iterVarName = _iterablePredicateVars[0];

		// evaluate from, to, incr only once (assumption: known at for entry)
		IntObject from = executePredicateInstructions( 1, _fromInstructions, ec );
		IntObject to   = executePredicateInstructions( 2, _toInstructions, ec );
		IntObject incr = executePredicateInstructions( 3, _incrementInstructions, ec );
		
		if ( incr.getIntValue() <= 0 ) //would produce infinite loop
			throw new DMLRuntimeException(this.printBlockErrorLocation() + "Expression for increment of variable '" + iterVarName + "' must evaluate to a positive value.");
		
		///////
		//OPTIMIZATION of ParFOR body (incl all child parfor PBs)
		///////
		if( _optMode != POptMode.NONE )
		{
			updateIterablePredicateVars( iterVarName, from, to, incr );
			OptimizationWrapper.optimize( _optMode, _sb, this, ec ); 
			
			//take changed iterable predicate into account
			iterVarName = _iterablePredicateVars[0];
			from = executePredicateInstructions( 1, _fromInstructions, ec );
			to   = executePredicateInstructions( 2, _toInstructions, ec );
			incr = executePredicateInstructions( 3, _incrementInstructions, ec );
		}
		
		///////
		//DATA PARTITIONING of read-only parent variables of type (matrix,unpartitioned)
		///////
		Timing time = null;
		if( MONITOR )
		{
			time = new Timing();
			time.start();
		}
		if( _dataPartitioner != PDataPartitioner.NONE )
		{			
			ArrayList<String> vars = (_sb!=null) ? _sb.getReadOnlyParentVars() : null;
			for( String var : vars )
			{
				Data dat = ec.getVariable(var);
				if( dat != null && dat instanceof MatrixObject )
				{
					MatrixObject moVar = (MatrixObject) dat;
					if( !moVar.isPartitioned() )
					{
						PDataPartitionFormat dpf = _sb.determineDataPartitionFormat( var );
						LOG.trace("PARFOR ID = "+_ID+", Partitioning read-only input variable "+var+" (format="+dpf+", mode="+_dataPartitioner+")");
						if( dpf != PDataPartitionFormat.NONE )
						{
							Timing ltime = new Timing();
							ltime.start();
							if( !ALLOW_UNSCOPED_PARTITIONING ) //store reference of original var
								_variablesDPOriginal.put(var, moVar);
							DataPartitioner dp = createDataPartitioner( dpf, _dataPartitioner );
							MatrixObject moVarNew = dp.createPartitionedMatrixObject(moVar);
							ec.setVariable(var, moVarNew);
							ProgramRecompiler.rFindAndRecompileIndexingHOP(_sb,this,var,ec);
							LOG.trace("Partitioning and recompilation done in "+ltime.stop()+"ms");
						}
					}
					else if( ALLOW_UNSCOPED_PARTITIONING ) //note: vars partitioned and not recompiled can only happen in case of unscoped partitioning over multiple top-level parfors.
					{
						//only recompile if input matrix is already partitioned.
						Timing ltime = new Timing();
						ltime.start();
						ProgramRecompiler.rFindAndRecompileIndexingHOP(_sb,this,var,ec);
						LOG.trace("Recompilation done in "+ltime.stop()+" ms");
					}
				}
			}
		}
		if( MONITOR ) 
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_INIT_DATA_T, time.stop());
			
		// initialize iter var to form value
		IntObject iterVar = new IntObject(iterVarName, from.getIntValue() );
		
		///////
		//begin PARALLEL EXECUTION of (PAR)FOR body
		///////
		LOG.trace("EXECUTE PARFOR ID = "+_ID+" with mode = "+_execMode+", numThreads = "+_numThreads+", taskpartitioner = "+_taskPartitioner);
		
		
		if( MONITOR )
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
					executeRemoteParFor(ec, iterVar, from, to, incr);
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
		iterVar = new IntObject( iterVarName, to.getIntValue() ); //consistent with for
		ec.setVariable(iterVarName, iterVar);
		
		//ensure that subsequent program blocks only see partitioned data if allowed
		if( !ALLOW_UNSCOPED_PARTITIONING )
		{
			//we can replace those variables, because partitioning only applied for read-only matrices
			for( String var : _variablesDPOriginal.keySet() )
			{
				MatrixObject mo = (MatrixObject) _variablesDPOriginal.get( var );
				ec.setVariable(var, mo);
			}
		}
		
		
		///////
		//end PARALLEL EXECUTION of (PAR)FOR body
		///////
			
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
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 * @throws InterruptedException 
	 */
	private void executeLocalParFor( ExecutionContext ec, IntObject itervar, IntObject from, IntObject to, IntObject incr ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException, InterruptedException
	{
		/* Step 1) init parallel workers, task queue and threads
		 *         start threads (from now on waiting for tasks)
		 * Step 2) create tasks
		 *         put tasks into queue
		 *         mark end of task input stream
		 * Step 3) join all threads (wait for finished work)
		 * Step 4) collect results from each parallel worker
		 */

		Timing time = null;
		if( MONITOR )
		{
			time = new Timing();
			time.start();
		}
		
		//restrict recompilation to thread local memory
		setMemoryBudget();
		
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
		
		if( MONITOR ) 
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_INIT_PARWRK_T, time.stop());
		
		// Step 2) create tasks 
		TaskPartitioner partitioner = createTaskPartitioner(from, to, incr);
		int numIterations = partitioner.getNumIterations();
		int numCreatedTasks = -1;
		if( USE_STREAMING_TASK_CREATION )
		{
			//put tasks into queue (parworker start work on first tasks while creating tasks) 
			numCreatedTasks = partitioner.createTasks(queue);		
		}
		else
		{
			Collection<Task> tasks = partitioner.createTasks();
			numCreatedTasks = tasks.size();
			
			// put tasks into queue
			for( Task t : tasks )
				queue.enqueueTask( t );
			
			// mark end of task input stream
			queue.closeInput();		
		}
		if( MONITOR )
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_INIT_TASKS_T, time.stop());
		
		// Step 3) join all threads (wait for finished work)
		for( Thread thread : threads )
			thread.join();
		
		if( MONITOR ) 
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_WAIT_EXEC_T, time.stop());
			
			
		// Step 4) collecting results from each parallel worker
		//obtain results
		LocalVariableMap [] localVariables = new LocalVariableMap [_numThreads]; 
		int numExecutedTasks = 0;
		int numExecutedIterations = 0;
		for( int i=0; i<_numThreads; i++ )
		{
			localVariables[i] = workers[i].getVariables();
			numExecutedTasks += workers[i].getExecutedTasks();
			numExecutedIterations += workers[i].getExecutedIterations();			
		}
		//consolidate results into global symbol table
		consolidateAndCheckResults( ec, numIterations, numCreatedTasks, numExecutedIterations, numExecutedTasks, 
				                    localVariables );
		
		//remove thread-local memory budget
		resetMemoryBudget();
		
		if( MONITOR ) 
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
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 * @throws IOException 
	 */
	private void executeRemoteParFor( ExecutionContext ec, IntObject itervar, IntObject from, IntObject to, IntObject incr ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException, IOException
	{
		/* Step 0) check and recompile MR inst
		 * Step 1) serialize child PB and inst
		 * Step 2) create tasks
		 *         serialize tasks
		 * Step 3) submit MR Jobs and wait for results                        
		 * Step 4) collect results from each parallel worker
		 */
		
		Timing time = null;
		if( MONITOR )
		{
			time = new Timing();
			time.start();
		}
		
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
		
		if( MONITOR ) 
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_INIT_PARWRK_T, time.stop());
		
		// Step 2) create tasks 
		TaskPartitioner partitioner = createTaskPartitioner(from, to, incr);
		String taskFile = constructTaskFileName();
		String resultFile = constructResultFileName();
		
		int numIterations = partitioner.getNumIterations();
		int maxDigits = (int)Math.log10(to.getIntValue()) + 1;
		int numCreatedTasks = -1;
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
			Collection<Task> tasks = partitioner.createTasks();
			numCreatedTasks        = tasks.size();
		    taskFile               = writeTasksToFile( taskFile, tasks, maxDigits );				
		}
				
		if( MONITOR )
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_INIT_TASKS_T, time.stop());
		
		//write matrices to HDFS 
		exportMatricesToHDFS(ec);
				
		// Step 3) submit MR job (wait for finished work)
		MatrixObject colocatedDPMatrixObj = (_colocatedDPMatrix!=null)? (MatrixObject)ec.getVariable(_colocatedDPMatrix) : null;
		RemoteParForJobReturn ret = RemoteParForMR.runJob(_ID, program, taskFile, resultFile, colocatedDPMatrixObj,
				                                          ExecMode.CLUSTER, _numThreads, WRITE_REPLICATION_FACTOR, MAX_RETRYS_ON_ERROR, getMinMemory(ec),
				                                          (ALLOW_REUSE_MR_JVMS & _jvmReuse) );
		
		if( MONITOR ) 
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_WAIT_EXEC_T, time.stop());
			
			
		// Step 4) collecting results from each parallel worker
		int numExecutedTasks = ret.getNumExecutedTasks();
		int numExecutedIterations = ret.getNumExecutedIterations();
		
		//consolidate results into global symbol table
		consolidateAndCheckResults( ec, numIterations, numCreatedTasks, numExecutedIterations , numExecutedTasks, 
				                    ret.getVariables() );
		if( flagForced ) //see step 0
			releaseForcedRecompile(0);
		
		if( MONITOR ) 
		{
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_WAIT_RESULTS_T, time.stop());
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_NUMTASKS, numExecutedTasks);
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_NUMITERS, numExecutedIterations);
		}			
	}	
	
	
	/**
	 * Cleanup result variables of parallel workers after result merge.
	 * @param in 
	 * @param out 
	 * @throws CacheException 
	 */
	private void cleanWorkerResultVariables(MatrixObject out, MatrixObject[] in) 
		throws CacheException
	{
		for( MatrixObject tmp : in )
		{
			//check for empty inputs (no iterations executed)
			if( tmp != null && tmp != out )
				tmp.clearData();
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
						}
					case MATRIX:
						//TODO currently we do not create any unscoped matrix object outputs
						//because metadata (e.g., outputinfo) not known at this place.
						break;
					case UNKNOWN:
						break;
					default:
						throw new DMLRuntimeException("Datatype not supported: "+datatype);
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
		if( LIVEVAR_AWARE_EXPORT && _sb != null)
		{
			//optimization to prevent unnecessary export of matrices
			//export only variables that are read in the body
			VariableSet varsRead = _sb.variablesRead();
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
		if( LIVEVAR_AWARE_CLEANUP && _sb != null)
		{
			//cleanup shared variables after they are unpinned
			VariableSet liveout = _sb.liveOut();
			for( Entry<String, Boolean> var : varState.entrySet() ) 
			{
				String varname = var.getKey();
				boolean unpinned = var.getValue();
				//delete unpinned vars if not in liveout (similar like rmvar)
				if( unpinned && !liveout.containsVariable(varname) )
				{
					VariableCPInstruction.processRemoveVariableInstruction(ec,varname);
				}
			}
		}
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
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 * @throws CloneNotSupportedException
	 */
	private LocalParWorker createParallelWorker(long pwID, LocalTaskQueue<Task> queue, ExecutionContext ec) 
		throws DMLRuntimeException
	{
		LocalParWorker pw = null; 
		
		try
		{
			//create deep copies of required elements
			//child blocks
			ArrayList<ProgramBlock> cpChildBlocks = null;		
			if( USE_PB_CACHE )
			{
				if( _pbcache.containsKey(pwID) )
				{
					cpChildBlocks = _pbcache.get(pwID);	
				}
				else
				{
					cpChildBlocks = ProgramConverter.rcreateDeepCopyProgramBlocks(_childBlocks, pwID, _IDPrefix, new HashSet<String>(), false); 
					_pbcache.put(pwID, cpChildBlocks);
				}
			}
			else
			{
				cpChildBlocks = ProgramConverter.rcreateDeepCopyProgramBlocks(_childBlocks, pwID, _IDPrefix, new HashSet<String>(), false); 
			}             
			
			// Deep copy Execution Context
			ExecutionContext cpEc = ProgramConverter.createDeepCopyExecutionContext(ec);
			
			//create the actual parallel worker
			ParForBody body = new ParForBody( cpChildBlocks, _resultVars, cpEc );
			pw = new LocalParWorker( pwID, queue, body, MAX_RETRYS_ON_ERROR );
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
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private DataPartitioner createDataPartitioner(PDataPartitionFormat dpf, PDataPartitioner dataPartitioner) 
		throws DMLRuntimeException 
	{
		DataPartitioner dp = null;
		
		switch( dataPartitioner )
		{
			case LOCAL:
				dp = new DataPartitionerLocal(dpf, -1, _numThreads);
				break;
			case REMOTE_MR:
				int numReducers = ConfigurationManager.getConfig().getIntValue(DMLConfig.NUM_REDUCERS);
				dp = new DataPartitionerRemoteMR( dpf, -1, _ID, 
						                          //Math.max(_numThreads,InfrastructureAnalyzer.getRemoteParallelMapTasks()), 
						                          Math.min(numReducers,InfrastructureAnalyzer.getRemoteParallelReduceTasks()),
						                          _replicationDP, 
						                          MAX_RETRYS_ON_ERROR, 
						                          ALLOW_REUSE_MR_JVMS );
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
	private ResultMerge createResultMerge( PResultMerge prm, MatrixObject out, MatrixObject[] in, String fname ) 
		throws DMLRuntimeException 
	{
		ResultMerge rm = null;
		
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
				int numReducers = ConfigurationManager.getConfig().getIntValue(DMLConfig.NUM_REDUCERS);
				rm = new ResultMergeRemoteMR( out, in, fname, _ID, 
					                          Math.max(_numThreads,InfrastructureAnalyzer.getRemoteParallelMapTasks()), 
					                          Math.min(numReducers,InfrastructureAnalyzer.getRemoteParallelReduceTasks()),
					                          WRITE_REPLICATION_FACTOR, 
					                          MAX_RETRYS_ON_ERROR, 
					                          ALLOW_REUSE_MR_JVMS );
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
		if( _sb == null ) {
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
	private String writeTasksToFile(String fname, Collection<Task> tasks, int maxDigits)
		throws DMLRuntimeException, IOException
	{
		BufferedWriter br = null;
		try
		{
			Path path = new Path(fname);
			FileSystem fs = FileSystem.get(new Configuration());
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
			FileSystem fs = FileSystem.get(new Configuration());
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
	private void consolidateAndCheckResults(ExecutionContext ec, int expIters, int expTasks, int numIters, int numTasks, LocalVariableMap [] results) 
		throws DMLRuntimeException
	{
		//result merge
		if( checkParallelRemoteResultMerge() )
		{
			//execute result merge in parallel for all result vars
			int par = Math.min( _resultVars.size(), 
					            InfrastructureAnalyzer.getLocalParallelism() );
			
			try
			{
				//enqueue all result vars as tasks
				LocalTaskQueue<String> q = new LocalTaskQueue<String>();
				for( String var : _resultVars ) //foreach non-local write
					q.enqueueTask(var);
				q.closeInput();
				
				//run result merge workers
				Thread[] rmWorkers = new Thread[par];
				for( int i=0; i<par; i++ )
					rmWorkers[i] = new Thread(new ResultMergeWorker(q, results, ec));
				for( int i=0; i<par; i++ ) //start all
					rmWorkers[i].start();
				for( int i=0; i<par; i++ ) //wait for all
					rmWorkers[i].join();
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
				String varname = var;
				MatrixObject out = (MatrixObject) ec.getVariable(varname);
				MatrixObject[] in = new MatrixObject[ results.length ];
				for( int i=0; i< results.length; i++ )
					in[i] = (MatrixObject) results[i].get( varname ); 			
				String fname = constructResultMergeFileName();
				ResultMerge rm = createResultMerge(_resultMerge, out, in, fname);
				MatrixObject outNew = null;
				if( USE_PARALLEL_RESULT_MERGE )
					outNew = rm.executeParallelMerge( _numThreads );
				else
					outNew = rm.executeSerialMerge(); 			
				ec.setVariable(varname, outNew);
		
				//cleanup of intermediate result variables
				cleanWorkerResultVariables( out, in );
			}
		}
		//handle unscoped variables (vars created in parfor, but potentially used afterwards)
		if( CREATE_UNSCOPED_RESULTVARS && _sb != null && ec.getVariables() != null ) //sb might be null for nested parallelism
			createEmptyUnscopedVariables( ec.getVariables(), _sb );
		
		//check expected counters
		if( numTasks != expTasks || numIters !=expIters ) //consistency check
			throw new DMLRuntimeException("PARFOR: Number of executed tasks does not match the number of created tasks: tasks "+numTasks+"/"+expTasks+", iters "+numIters+"/"+expIters+".");
	}
	
	/**
	 * NOTE: Currently we use a fixed rule (multiple results AND REMOTE_MR -> only selected by the optimizer
	 * if mode was REMOTE_MR as well). 
	 * TODO Eventually, the optimizer should decide about parallel result merge and its degree of parallelism.
	 * 
	 * @return
	 */
	private boolean checkParallelRemoteResultMerge()
	{
		return (USE_PARALLEL_RESULT_MERGE_REMOTE 
			    && _resultVars.size() > 1
			    && _resultMerge == PResultMerge.REMOTE_MR);
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
	 * 
	 */
	private void setLocalParWorkerIDs()
	{
		//set all parworker IDs required if PExecMode.LOCAL is used
		_pwIDs = new long[ _numThreads ];
		
		for( int i=0; i<_numThreads; i++ )
		{
			if(_IDPrefix == -1)
				_pwIDs[i] = _pwIDSeq.getNextID();
			else
				_pwIDs[i] = IDHandler.concatIntIDsToLong(_IDPrefix,(int)_pwIDSeq.getNextID());
			
			if(MONITOR ) 
				StatisticMonitor.putPfPwMapping(_ID, _pwIDs[i]);
		}
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
		_numIterations = (int)Math.ceil(((double)(to.getIntValue() - from.getIntValue() + 1)) / incr.getIntValue()); 
		
		_iterablePredicateVars[0] = iterVarName;
		_iterablePredicateVars[1] = from.getStringValue();
		_iterablePredicateVars[2] = to.getStringValue();
		_iterablePredicateVars[3] = incr.getStringValue();
	}
	
	/**
	 * NOTE: Only required for remote parfor. Hence, there is no need to transfer DMLConfig to
	 * the remote workers (MR job) since nested remote parfor is not supported.
 	 * 
	 * @return
	 */
	private String constructTaskFileName()
	{
		String scratchSpaceLoc = ConfigurationManager.getConfig()
        							.getTextValue(DMLConfig.SCRATCH_SPACE);
	
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
		String scratchSpaceLoc = ConfigurationManager.getConfig()
									.getTextValue(DMLConfig.SCRATCH_SPACE);
		
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
		String scratchSpaceLoc = ConfigurationManager.getConfig()
		                             .getTextValue(DMLConfig.SCRATCH_SPACE);
		
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
				OptTree tree;
				tree = OptTreeConverter.createAbstractOptTree(-1, -1, _sb, this, new HashSet<String>(), ec);
				CostEstimator est = new CostEstimatorHops( OptTreeConverter.getAbstractPlanMapping() );
				double mem = est.getEstimate(TestMeasure.MEMORY_USAGE, tree.getRoot());
				
				ret = (long) (mem * ( 1d/OptimizerUtils.MEM_UTIL_FACTOR  )); 
			} 
			catch(Exception e) 
			{
				e.printStackTrace();
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
	
	
	public String printBlockErrorLocation(){
		return "ERROR: Runtime error in parfor program block generated from parfor statement block between lines " + _beginLine + " and " + _endLine + " -- ";
	}
	
	
	/**
	 * Helper class for parallel invocation of REMOTE_MR result merge for multiple variables.
	 */
	private class ResultMergeWorker implements Runnable
	{
		private LocalTaskQueue<String> _q = null;
		private LocalVariableMap[] _refVars = null;
		private ExecutionContext _ec = null;
		
		public ResultMergeWorker( LocalTaskQueue<String> q, LocalVariableMap[] results, ExecutionContext ec )
		{
			_q = q;
			_refVars = results;
			_ec = ec;
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
				
					ResultMerge rm = createResultMerge(_resultMerge, out, in, fname);
					MatrixObject outNew = null;
					if( USE_PARALLEL_RESULT_MERGE )
						outNew = rm.executeParallelMerge( _numThreads );
					else
						outNew = rm.executeSerialMerge(); 	
					
					synchronized( _ec.getVariables() ){
						_ec.getVariables().put( varname, outNew);
					}
		
					//cleanup of intermediate result variables
					cleanWorkerResultVariables( out, in );
				}
			}
			catch(Exception ex)
			{
				LOG.error("Error executing result merge: ", ex);
			}
		}
	}
	
}