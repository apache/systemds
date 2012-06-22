package com.ibm.bi.dml.runtime.controlprogram;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.parser.ParForStatementBlock;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.LocalParWorker;
import com.ibm.bi.dml.runtime.controlprogram.parfor.LocalTaskQueue;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ParForBody;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.RemoteParForJobReturn;
import com.ibm.bi.dml.runtime.controlprogram.parfor.RemoteParForMR;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ResultMerge;
import com.ibm.bi.dml.runtime.controlprogram.parfor.Task;
import com.ibm.bi.dml.runtime.controlprogram.parfor.TaskPartitioner;
import com.ibm.bi.dml.runtime.controlprogram.parfor.TaskPartitionerFactoring;
import com.ibm.bi.dml.runtime.controlprogram.parfor.TaskPartitionerFactoringConstrained;
import com.ibm.bi.dml.runtime.controlprogram.parfor.TaskPartitionerFixedsize;
import com.ibm.bi.dml.runtime.controlprogram.parfor.TaskPartitionerNaive;
import com.ibm.bi.dml.runtime.controlprogram.parfor.TaskPartitionerStatic;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Stat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.StatisticMonitor;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Timing;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDHandler;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObjectNew;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionContext;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;

/**
 * The ParForProgramBlock has the same execution semantics as a ForProgamBlock but executes
 * the independent iterations in parallel. See ParForStatementBlock for the loop dependency
 * analysis. At runtime level, iterations are guaranteed to be completely independent.
 * 
 * 
 * @author mboehm
 */
public class ParForProgramBlock extends ForProgramBlock 
{
	// execution modes
	public enum PExecMode
	{
		LOCAL,      //local (master) multi-core execution mode
		REMOTE_MR,	//remote (MR cluster) execution mode	
		
		@Deprecated
		REMOTE_MR_STREAM //remote (MR cluster) execution mode via MR Streaming	
	}

	// task partitioner
	public enum PTaskPartitioner
	{
		FIXED,      //fixed-sized task partitioner, uses tasksize 
		NAIVE,      //naive task partitioner (tasksize=1)
		STATIC,     //static task partitioner (numIterations/numThreads)
		FACTORING,  //factoring task partitioner  
		CFACTORING  //constrained factoring task partitioner, uses tasksize as min constraint
	}
	
	//optimizer
	public enum POptMode
	{
		NONE,       //no optimization, use defaults and specified parameters
		FULL,       //
		FULL_DP,    //
		GREEDY,     //
		GREEDY_PARTIAL; //		
	}

		
	// internal parameters
	public static final boolean MONITOR                     = false;	// collect internal statistics
	public static final boolean OPTIMIZE                    = false;	// run all automatic optimizations on top-level parfor
	public static final boolean USE_PB_CACHE                = true;  	// reuse copied program blocks whenever possible
	public static       boolean USE_RANGE_TASKS_IF_USEFUL   = false;   	// use range tasks whenever size>3
	public static final boolean USE_STREAMING_TASK_CREATION = true;  	// start working while still creating tasks
	public static final boolean USE_BINARY_MR_TASK_REP	    = false;    // serialize tasks to binary representation for remote communication
	public static final boolean ALLOW_NESTED_PARALLELISM	= true;    // if not, transparently change parfor to for on program conversions (local,remote)
	public static final boolean ALLOW_REUSE_MR_JVMS         = true;     // potential benefits: less setup costs per task
	public static final boolean ALLOW_REUSE_MR_PAR_WORKER   = ALLOW_REUSE_MR_JVMS; //potential benefits: less initialization, reuse in-memory objects and result consolidation!
	public static final boolean USE_FLEX_SCHEDULER_CONF     = false;
	public static final boolean WRITE_RESULTS_TO_FILE       = true;
	public static final int     WRITE_REPLICATION_FACTOR    = 3;
	public static final int     MAX_RETRYS_ON_ERROR         = 0;
	
	public static final String PARFOR_MR_TASKS_TMP_FNAME    = "/parfor/%ID%_taskfile.dat"; 
	public static final String PARFOR_MR_RESULT_TMP_FNAME   = "/parfor/%ID%_results.dat"; 
	
	// static ID generator sequences
	private static IDSequence   _pfIDSeq        = null;
	private static IDSequence   _pwIDSeq        = null;
	
	// runtime parameters
	protected HashMap<String,String> _params    = null;
	protected int              _numThreads      = -1;
	protected int              _taskSize        = -1;
	protected PTaskPartitioner _taskPartitioner = null; 
	protected PExecMode        _execMode        = null;
	protected POptMode         _optMode         = null;
	
	// program block meta data
	protected long                _ID           = -1;
	protected int                 _IDPrefix     = -1;
	protected ArrayList<String>   _resultVars   = null;
	
	// local parworker data
	protected long[] 		   	                     _pwIDs   = null;
	protected HashMap<Long,ArrayList<ProgramBlock>>  _pbcache = null;

	
	static
	{
		//init static ID sequence generators
		_pfIDSeq = new IDSequence();
		_pwIDSeq = new IDSequence();
	}
	
	public ParForProgramBlock(Program prog, String[] iterPredVars, HashMap<String,String> params) throws DMLRuntimeException 
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
	public ParForProgramBlock(int ID, Program prog, String[] iterPredVars, HashMap<String,String> params) throws DMLRuntimeException  
	{
		super(prog, iterPredVars);
	
		//ID generation and setting 
		setParForProgramBlockIDs( ID );
		
		//parse and use internal parameters (already set to default if not specified)
		_params = params;
		try
		{
			_numThreads      = Integer.parseInt( _params.get(ParForStatementBlock.PAR) );
			_taskSize        = Integer.parseInt( _params.get(ParForStatementBlock.TASK_SIZE) );
			_taskPartitioner = PTaskPartitioner.valueOf( _params.get(ParForStatementBlock.TASK_PARTITIONER).toUpperCase() );
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
			
		//create IDs for all parworkers
		if( _execMode == PExecMode.LOCAL )
			setLocalParWorkerIDs();
	
		//initialize program block cache if necessary
		if( USE_PB_CACHE ) 
			_pbcache = new HashMap<Long, ArrayList<ProgramBlock>>();
		
		if( DMLScript.DEBUG )
			System.out.println("PARFOR: ParForProgramBlock created with numThreads="+_numThreads+", taskSize="+_taskSize);
	}
	
	public long getID()
	{
		return _ID;
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
			throw new DMLRuntimeException("Expression for increment of variable '" + iterVarName + "' must evaluate to a positive value.");
				
		// initialize iter var to form value
		IntObject iterVar = new IntObject(iterVarName, from.getIntValue() );

		///////
		//OPTIMIZATION of ParFOR body (incl all child parfor PBs)
		///////
		//TODO: decomment 
		//if( _optMode != POptMode.NONE )
		//	OptimizationWrapper.optimize( _optMode, this );
		
		
		///////
		//begin PARALLEL EXECUTION of (PAR)FOR body
		///////
		
		if( MONITOR )
		{
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_NUMTHREADS,      _numThreads);
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_TASKSIZE,        _taskSize);
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_TASKPARTITIONER, _taskPartitioner.ordinal());
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_EXECMODE,        _execMode.ordinal());
		}
		
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
				
				case REMOTE_MR_STREAM:
					throw new DMLRuntimeException("ParFOR: exec mode REMOTE_MR_STREAM not supported yet. Use REMOTE_MR.");
					//executeRemoteParFor2(ec, iterVar, from, to, incr);
					//break;
					
				default:
					throw new DMLRuntimeException("Undefined execution mode: '"+_execMode+"'.");
			}	
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("PARFOR: Failed to execute loop in parallel.",ex);
		}
		
		//set iteration var to TO value (+ increment) for FOR equivalence
		iterVar = new IntObject( iterVarName, to.getIntValue() ); //consistent with for
		_variables.put(iterVarName, iterVar);
		
		///////
		//end PARALLEL EXECUTION of (PAR)FOR body
		///////
			
		execute(_exitInstructions, ec);			
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
	@SuppressWarnings("unchecked")
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
		
		// Step 1) init parallel workers, task queue and threads
		LocalTaskQueue queue     = new LocalTaskQueue();
		Thread[] threads         = new Thread[_numThreads];
		LocalParWorker[] workers = new LocalParWorker[_numThreads];
		for( int i=0; i<_numThreads; i++ )
		{
			//create parallel workers as (lazy) deep copies
			workers[i] = createParallelWorker( _pwIDs[i], queue, ec ); 
			threads[i] = new Thread( workers[i] );
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
		//concolidate results into global symbol table
		consolidateAndCheckResults( numIterations, numCreatedTasks, numExecutedIterations, numExecutedTasks, 
				                    localVariables );
		
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
		/* Step 1) serialize child PB and inst
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
		
		// Step 1) init parallel workers (serialize PBs)
		// NOTES: each mapper changes filenames with regard to his ID as we submit a single job,
		//        cannot reuse serialized string, since variables are serialized as well.
		ParForBody body = new ParForBody( _childBlocks, _variables, _resultVars, ec );
		String program = ProgramConverter.serializeParForBody( body );
		
		if( MONITOR ) 
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_INIT_PARWRK_T, time.stop());
		
		// Step 2) create tasks 
		TaskPartitioner partitioner = createTaskPartitioner(from, to, incr);
		String taskFile = constructTaskfileName();
		String resultFile = constructResultfileName();

		int numIterations = partitioner.getNumIterations();
		int numCreatedTasks = -1;
		if( USE_STREAMING_TASK_CREATION )
		{
			LocalTaskQueue queue = new LocalTaskQueue();
		
			//put tasks into queue and start writing to taskFile
			numCreatedTasks = partitioner.createTasks(queue);
			taskFile        = writeTasksToFile( taskFile, queue );				
		}
		else
		{
			//sequentially create tasks and write to disk
			Collection<Task> tasks = partitioner.createTasks();
			numCreatedTasks        = tasks.size();
		    taskFile               = writeTasksToFile( taskFile, tasks );				
		}
				
		if( MONITOR )
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_INIT_TASKS_T, time.stop());
		
		// TODO Matthias: transfer only required!
		// TODO: check getVarible()
		
		/* if there is some data in the cache that is not backed by HDFS 
		 * (i.e., a "dirty" matrix object), then it must be exported prior
		 * to serialization.
		 */
		for (String key : _variables.keySet() ) {
			Data d = getVariable(key);
			if ( d.getDataType() == DataType.MATRIX ) {
				if ( ((MatrixObjectNew)d).isDirty() )
					((MatrixObjectNew)d).exportData();
			}
		}
				
		// Step 3) submit MR job (wait for finished work)
		RemoteParForJobReturn ret = RemoteParForMR.runJob(_ID, program, taskFile, resultFile, 
				                      ExecMode.CLUSTER, _numThreads, WRITE_REPLICATION_FACTOR, MAX_RETRYS_ON_ERROR);
		
		if( MONITOR ) 
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_WAIT_EXEC_T, time.stop());
			
			
		// Step 4) collecting results from each parallel worker
		int numExecutedTasks = ret.getNumExecutedTasks();
		int numExecutedIterations = ret.getNumExecutedIterations();
		
		//concolidate results into global symbol table
		consolidateAndCheckResults( numIterations, numCreatedTasks, numExecutedIterations , numExecutedTasks, 
				                    ret.getVariables() );

		if( MONITOR ) 
		{
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_WAIT_RESULTS_T, time.stop());
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_NUMTASKS, numExecutedTasks);
			StatisticMonitor.putPFStat(_ID, Stat.PARFOR_NUMITERS, numExecutedIterations);
		}			
	}	
	
	/**
	 * 
	 * @param q
	 * @param ec
	 * @param itervar
	 * @param from
	 * @param to
	 * @param incr
	 * @throws InterruptedException
	 * @throws DMLRuntimeException
	 * @throws IOException
	 * @throws DMLUnsupportedOperationException 
	 */
	private void migrateExecutionFromLocalToRemote(LocalTaskQueue q, ExecutionContext ec, IntObject itervar, IntObject from, IntObject to, IntObject incr) 
		throws DMLRuntimeException, DMLUnsupportedOperationException, InterruptedException, IOException
	{
		// Step 1) create remote ParWorkers
		ParForBody body= new ParForBody(_childBlocks, _variables, _resultVars, ec);
		String program = ProgramConverter.serializeParForBody( body );
		
		// Step 2) obtain remaining tasks
		String taskFile = constructTaskfileName();
		if( USE_STREAMING_TASK_CREATION )
		{
			//write available tasks to task file
			taskFile = writeTasksToFile( taskFile, q );	
		}
		else
		{
			//grab all remaining tasks
			LinkedList<Task> tasks = new LinkedList<Task>();
			Task lTask = null;
			while( (lTask = q.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS )
				tasks.addLast( lTask );
			//write task file
			taskFile = writeTasksToFile( taskFile, tasks );		
		}
		
		// Step 3) submit MR job (wait for finished work)
		String resultFile = constructResultfileName();
		
		RemoteParForJobReturn ret = RemoteParForMR.runJob(_ID, program, taskFile, resultFile, 
				                                ExecMode.CLUSTER, _numThreads, WRITE_REPLICATION_FACTOR, MAX_RETRYS_ON_ERROR); 	
		
		// Step 4) collecting results from each parallel worker

		//concolidate results into global symbol table
		consolidateAndCheckResults( -1, -1, -1 , -1, //distributed over local and remote  
				                    ret.getVariables() );
	}
	
	/**
	 * Creates an new or partially recycled instance of a parallel worker. Therefore the symbol table, and child
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
	@SuppressWarnings("unchecked") 
	private LocalParWorker createParallelWorker(long pwID, LocalTaskQueue queue, ExecutionContext ec) 
		throws DMLRuntimeException
	{
		LocalParWorker pw = null; 
		
		try
		{
			//create deep copies of required elements
			//childblocks
			ArrayList<ProgramBlock> cpChildBlocks = null;		
			if( USE_PB_CACHE )
			{
				if( _pbcache.containsKey(pwID) )
				{
					cpChildBlocks = _pbcache.get(pwID);	
				}
				else
				{
					cpChildBlocks = (ArrayList<ProgramBlock>) ProgramConverter.rcreateDeepCopyProgramBlocks(_childBlocks, pwID, _IDPrefix); 
					_pbcache.put(pwID, cpChildBlocks);
				}
			}
			else
			{
				cpChildBlocks = (ArrayList<ProgramBlock>) ProgramConverter.rcreateDeepCopyProgramBlocks(_childBlocks, pwID, _IDPrefix); 
			}                     
			//symbol table
			LocalVariableMap cpVars = (LocalVariableMap) _variables.clone();
			
			//create the actual parallel worker
			ParForBody body = new ParForBody( cpChildBlocks, cpVars, _resultVars, ec );
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
			case CFACTORING:
				//for constrained factoring the tasksize is used as the minimum constraint
				tp = new TaskPartitionerFactoringConstrained( _taskSize,_numThreads, _taskSize, _iterablePredicateVars[0],
							                       from, to, incr );
				break;
			default:
				throw new DMLRuntimeException("Undefined task partitioner: '"+_taskPartitioner+"'.");
		}
		
		return tp;
	}
	
	/**
	 * 
	 * @param fname
	 * @param tasks
	 * @return
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
	private String writeTasksToFile(String fname, Collection<Task> tasks)
		throws DMLRuntimeException, IOException
	{
		BufferedWriter br = null;
		try
		{
			Path path = new Path(fname);
			FileSystem fs = FileSystem.get(new Configuration());
			br = new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));
	        
			for( Task t : tasks )
			{
				if( USE_BINARY_MR_TASK_REP )
					br.write( t.toBinary() + "\n" );
				else
					br.write( t.toCompactString() + "\n" );
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
	private String writeTasksToFile(String fname, LocalTaskQueue queue)
		throws DMLRuntimeException, IOException
	{
		BufferedWriter br = null;
		try
		{
			Path path = new Path( fname );
			FileSystem fs = FileSystem.get(new Configuration());
			br = new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));
	        
			Task t = null;
			while( (t = queue.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS )
			{
				if( USE_BINARY_MR_TASK_REP )
					br.write( t.toBinary() + "\n" );
				else
					br.write( t.toCompactString() + "\n" );
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
	 * @param expIters
	 * @param expTasks
	 * @param numIters
	 * @param numTasks
	 * @param results
	 * @throws DMLRuntimeException
	 */
	private void consolidateAndCheckResults(int expIters, int expTasks, int numIters, int numTasks, LocalVariableMap [] results) 
		throws DMLRuntimeException
	{
		for( String var : _resultVars ) //foreach non-local write
		{
			String varname = var;
			
			/*
			 * TODO Matthias
			 * 1) make necessary changes to acquire appropriate locks on in, out
			 * 2) Read existing values for 'out' before performing the merge
			 * 3) pass the MatrixObjectNew(out).getData() to the merge method
			 *  
			 */
			MatrixObjectNew out = (MatrixObjectNew) getVariable(varname);
			MatrixObjectNew[] in = new MatrixObjectNew[ results.length ];
			for( int i=0; i< results.length; i++ )
				in[i] = (MatrixObjectNew) results[i].get( varname ); 
			ResultMerge rm = new ResultMerge(out,in, WRITE_RESULTS_TO_FILE, WRITE_RESULTS_TO_FILE);
			_variables.put( varname, rm.executeParallelMerge( _numThreads ));
		}
		
		if( numTasks != expTasks || numIters !=expIters ) //consistency check
			throw new DMLRuntimeException("PARFOR: Number of executed tasks does not match the number of created tasks: tasks "+numTasks+"/"+expTasks+", iters "+numIters+"/"+expIters+".");
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
	 * @return
	 */
	private String constructTaskfileName()
	{
		return ConfigurationManager.getConfig().getTextValue("scratch")+
		       PARFOR_MR_TASKS_TMP_FNAME.replaceAll("%ID%", String.valueOf(_ID));   
	}
	
	/**
	 * 
	 * @return
	 */
	private String constructResultfileName()
	{
		return ConfigurationManager.getConfig().getTextValue("scratch")+
		       PARFOR_MR_RESULT_TMP_FNAME.replaceAll("%ID%", String.valueOf(_ID));   
	}
}