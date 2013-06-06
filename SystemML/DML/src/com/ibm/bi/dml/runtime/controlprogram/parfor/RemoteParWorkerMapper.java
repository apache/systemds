package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.IOException;
import java.util.HashMap;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheStatistics;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheableData;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Stat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.StatisticMonitor;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDHandler;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;

/**
 * Remote ParWorker implementation, realized as MR mapper.
 * 
 *
 */
public class RemoteParWorkerMapper extends ParWorker  //MapReduceBase not required (no op implementations of configure, close)
	implements Mapper<LongWritable, Text, Writable, Writable>
{
	//cache for future reuse (in case of JVM reuse)
	//NOTE: Hashmap to support multiple parfor MR jobs for local mode and if JVM reuse across jobs
	private static HashMap<String,RemoteParWorkerMapper> _sCache = null; 
	
	//MR ParWorker attributes  
	protected String  _stringID       = null; 
	
	static
	{
		//init cache (once per JVM)
		_sCache = new HashMap<String, RemoteParWorkerMapper>();
	}
	
	
	public RemoteParWorkerMapper( ) 
	{
		
	}
	
	/**
	 * 
	 */
	public void map(LongWritable key, Text value, OutputCollector<Writable, Writable> out, Reporter reporter) 
		throws IOException
	{
		LOG.trace("execute RemoteParWorkerMapper "+_stringID+" ("+_workerID+")");
		
		int numIters = getExecutedIterations(); //for multiple iterations
		
		try 
		{
			//parse input task
			Task lTask = Task.parseCompactString( value.toString() );
			
			//execute task (on error: re-try via Hadoop)
			executeTask( lTask );
		
			//write output if required (matrix indexed write)
			LongWritable okey = new LongWritable( _workerID ); //created once
			Text ovalue = new Text();
			for( String rvar : _resultVars )
			{
				Data dat = _ec.getSymbolTable().getVariable(rvar );
				
				//export output variable to HDFS (see RunMRJobs)
				if ( dat.getDataType() == DataType.MATRIX ) {
					MatrixObject inputObj = (MatrixObject) dat;
					inputObj.exportData(); //note: this is equivalent to doing it in close (currently not required because 1 Task=1Map tasks, hence only one map invocation)
				}
				
				//pass output vars (scalars by value, matrix by ref) to result
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
		
		if( CacheableData.CACHING_STATS  && !InfrastructureAnalyzer.isLocalMode() )
		{
			reporter.incrCounter( CacheStatistics.Stat.CACHE_HITS_MEM, CacheStatistics.getMemHits());
			reporter.incrCounter( CacheStatistics.Stat.CACHE_HITS_FS, CacheStatistics.getFSHits());
			reporter.incrCounter( CacheStatistics.Stat.CACHE_HITS_HDFS, CacheStatistics.getHDFSHits());
			reporter.incrCounter( CacheStatistics.Stat.CACHE_WRITES_FS, CacheStatistics.getFSWrites());
			reporter.incrCounter( CacheStatistics.Stat.CACHE_WRITES_HDFS, CacheStatistics.getHDFSWrites());
		}
	}

	/**
	 * 
	 */
	public void configure(JobConf job)
	{
		boolean requiresConfigure = true;
		String jobID = job.get("mapred.job.id");
		
		//probe cache for existing worker (parfor body, symbol table, etc)
		if( ParForProgramBlock.ALLOW_REUSE_MR_PAR_WORKER )
		{
			synchronized( _sCache ) //for multiple jobs in local mode
			{
				if( _sCache.containsKey(jobID) )
				{
					RemoteParWorkerMapper tmp = _sCache.get(jobID);
					
					_stringID       = tmp._stringID;
					_workerID       = tmp._workerID;
					
					_childBlocks    = tmp._childBlocks;
					_resultVars     = tmp._resultVars;
					_ec             = tmp._ec;
					
					_numIters       = tmp._numIters;
					_numTasks       = tmp._numTasks;
										
					requiresConfigure = false;
				}
			}
		}
		
		if( requiresConfigure )
		{
			LOG.trace("configure RemoteParWorkerMapper "+job.get("mapred.tip.id"));
			
			try
			{
				//_stringID = job.get("mapred.task.id"); //task attempt ID
				_stringID = job.get("mapred.tip.id"); //task ID
				_workerID = IDHandler.extractIntID(_stringID); //int task ID
				
				//create local runtime program
				String in = MRJobConfiguration.getProgramBlocksInMapper(job);
				ParForBody body = ProgramConverter.parseParForBody(in, (int)_workerID);
				
				_childBlocks = body.getChildBlocks();
				_ec          = body.getEc();				
				_resultVars  = body.getResultVarNames();
		
				//init local cache manager 
				if( !CacheableData.isCachingActive() ) 
				{
					String uuid = IDHandler.createDistributedUniqueID();
					LocalFileUtils.createWorkingDirectoryWithUUID( uuid );
					CacheableData.initCaching( uuid ); //incl activation, cache dir creation (each map task gets its own dir for simplified cleanup)
				}
				
				if( !CacheableData.cacheEvictionLocalFilePrefix.contains("_") ) //account for local mode
				{
					CacheableData.cacheEvictionLocalFilePrefix = CacheableData.cacheEvictionLocalFilePrefix +"_" + _workerID; 
					CacheableData.cacheEvictionHDFSFilePrefix = CacheableData.cacheEvictionHDFSFilePrefix +"_" + _workerID;
				}
				
				//ensure that resultvar files are not removed
				pinResultVariables();
				
				_numTasks    = 0;
				_numIters    = 0;
				
			}
			catch(Exception ex)
			{
				throw new RuntimeException(ex);
			}
			
			//disable stat monitoring, reporting execution times via counters not useful 
			StatisticMonitor.disableStatMonitoring();
			
			//put into cache if required
			if( ParForProgramBlock.ALLOW_REUSE_MR_PAR_WORKER )
				synchronized( _sCache ){ //for multiple jobs in local mode
					_sCache.put(jobID, this);
				}
		} 
		else
		{
			LOG.trace("reuse configured RemoteParWorkerMapper "+_stringID);
		}
		
		//always reset cache stats because counters per map task
		if( CacheableData.CACHING_STATS && !InfrastructureAnalyzer.isLocalMode() )
			CacheStatistics.reset();
	}
	
	private void pinResultVariables()
	{
		for( String var : _resultVars )
		{
			Data dat = _ec.getSymbolTable().getVariable(var);
			if( dat instanceof MatrixObject )
			{
				MatrixObject mo = (MatrixObject)dat;
				mo.enableCleanup(false); 
			}
		}
	}

	/**
	 * 
	 */
	@Override
	public void close() throws IOException 
	{
		//cleanup cached variables in order to prevent writing to disk
		boolean isLocal = InfrastructureAnalyzer.isLocalMode();
		if( !isLocal && !ParForProgramBlock.ALLOW_REUSE_MR_PAR_WORKER )
		{
			CacheableData.cleanupCacheDir();
			CacheableData.disableCaching();
			LocalFileUtils.cleanupWorkingDirectory();
		}
	}
	
}
