package dml.runtime.controlprogram.parfor;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import dml.api.DMLScript;
import dml.runtime.controlprogram.ParForProgramBlock;
import dml.runtime.controlprogram.parfor.stat.Stat;
import dml.runtime.controlprogram.parfor.stat.StatisticMonitor;
import dml.runtime.controlprogram.parfor.util.IDHandler;
import dml.runtime.instructions.CPInstructions.Data;
import dml.runtime.matrix.mapred.MRJobConfiguration;

/**
 * Remote ParWorker implementation, realized as MR mapper.
 * 
 * @author mboehm
 *
 */
public class RemoteParWorkerMapper extends ParWorker  //MapReduceBase not required (no op implementations of configure, close)
	implements Mapper<LongWritable, Text, Writable, Writable>
{
	private static RemoteParWorkerMapper _sCache = null; //self reference for future reuse
	
	protected String            _stringID       = null; 
	protected boolean           _binaryTasks    = false;	
	protected ArrayList<String> _resultVarNames = null;
	
	
	public RemoteParWorkerMapper( ) 
	{
		
	}
	
	/**
	 * 
	 */
	public void map(LongWritable key, Text value, OutputCollector<Writable, Writable> out, Reporter reporter) 
		throws IOException
	{
		if( DMLScript.DEBUG )
			System.out.println("execute RemoteParWorkerMapper "+_stringID+" ("+_workerID+")");
		
		int numIters = getExecutedIterations(); //for multiple iterations
		
		try 
		{
			//parse input task
			Task lTask = null;
			if( _binaryTasks )
				lTask = Task.parseBinary( value.getBytes() );
			else
				lTask = Task.parseCompactString( value.toString() );
			
			//execute task (on error: re-try via Hadoop)
			executeTask( lTask );
		
			//write output if required (matrix indexed write)
			LongWritable okey = new LongWritable( _workerID ); //created once
			Text ovalue = new Text();
			for( String rvar : _resultVarNames )
			{
				Data dat = _variables.get( rvar );
				String datStr = ProgramConverter.serializeDataObject(rvar, dat);
				ovalue.set( datStr );
				out.collect( okey, ovalue );	
			}
		}
		catch(Exception ex)
		{
			//throw IO exception to adhere to API specification
			throw new IOException("ParFOR: Failed to execute task.",ex);
		}
		
		//statistic maintenance
		reporter.incrCounter(Stat.PARFOR_NUMITERS, getExecutedIterations()-numIters);
		reporter.incrCounter(Stat.PARFOR_NUMTASKS, 1);
	}

	/**
	 * 
	 */
	public void configure(JobConf job)
	{
		boolean requiresConfigure = true;
		
		if( ParForProgramBlock.ALLOW_REUSE_MR_PAR_WORKER )
		{
			synchronized( ParForProgramBlock.class )
			{
				if( _sCache != null )
				{
					_stringID       = _sCache._stringID;
					_workerID       = _sCache._workerID;
					_binaryTasks    = _sCache._binaryTasks;
					
					_childBlocks    = _sCache._childBlocks;
					_variables      = _sCache._variables;
					_resultVarNames = _sCache._resultVarNames;
					_ec             = _sCache._ec;
					
					_numIters       = _sCache._numIters;
					_numTasks       = _sCache._numTasks;
										
					requiresConfigure = false;
				}
			}
		}
		
		if( requiresConfigure )
		{
			if( DMLScript.DEBUG )
				System.out.println("configure RemoteParWorkerMapper "+job.get("mapred.tip.id"));
			
			try
			{
				//_stringID = job.get("mapred.task.id"); //task attempt ID
				_stringID = job.get("mapred.tip.id"); //task ID
				_workerID = IDHandler.extractIntID(_stringID); //int task ID
				
				String in = MRJobConfiguration.getProgramBlocksInMapper(job);
				
				if(DMLScript.DEBUG)
					System.out.println(in);
				
				ParForBody body = ProgramConverter.parseParForBody(in, (int)_workerID);
				
				_childBlocks = body.getChildBlocks();
				_variables   = body.getVariables();
				_ec          = body.getEc();
				
				_resultVarNames = body.getResultVarNames();
				
				_numTasks    = 0;
				_numIters    = 0;
				
			}
			catch(Exception ex)
			{
				throw new RuntimeException(ex);
			}
	
			//set task encoding
			_binaryTasks = ParForProgramBlock.USE_BINARY_MR_TASK_REP;
			
			//disable stat monitoring, reporting execution times via counters not useful 
			StatisticMonitor.disableStatMonitoring();
			
			if( ParForProgramBlock.ALLOW_REUSE_MR_PAR_WORKER )
				_sCache = this;
		} 
		else
		{
			if( DMLScript.DEBUG )
				System.out.println("reuse configuration RemoteParWorkerMapper "+_stringID);
		}
	}

	/**
	 * 
	 */
	@Override
	public void close() throws IOException 
	{
		//do nothing
	}
	
}
