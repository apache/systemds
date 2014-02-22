/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

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

import com.ibm.bi.dml.api.DMLScript;
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
import com.ibm.bi.dml.utils.Statistics;

/**
 * Remote ParWorker implementation, realized as MR mapper.
 * 
 * NOTE: In a cluster setup, reuse jvm will not lead to reusing jvms of different jobs or different
 * task types due to job-level specification of jvm max sizes for map/reduce 
 *
 */
public class RemoteParWorkerMapper extends ParWorker  //MapReduceBase not required (no op implementations of configure, close)
	implements Mapper<LongWritable, Text, Writable, Writable>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
				Data dat = _ec.getVariable( rvar );
				
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
		reporter.incrCounter(ParForProgramBlock.PARFOR_COUNTER_GROUP_NAME, Stat.PARFOR_NUMITERS.toString(), getExecutedIterations()-numIters);
		reporter.incrCounter(ParForProgramBlock.PARFOR_COUNTER_GROUP_NAME, Stat.PARFOR_NUMTASKS.toString(), 1);
		if( DMLScript.STATISTICS  && !InfrastructureAnalyzer.isLocalMode() ) {
			reporter.incrCounter( ParForProgramBlock.PARFOR_COUNTER_GROUP_NAME, Stat.PARFOR_JITCOMPILE.toString(), Statistics.getJITCompileTime());
			reporter.incrCounter( CacheableData.CACHING_COUNTER_GROUP_NAME, CacheStatistics.Stat.CACHE_HITS_MEM.toString(), CacheStatistics.getMemHits());
			reporter.incrCounter( CacheableData.CACHING_COUNTER_GROUP_NAME, CacheStatistics.Stat.CACHE_HITS_FSBUFF.toString(), CacheStatistics.getFSBuffHits());
			reporter.incrCounter( CacheableData.CACHING_COUNTER_GROUP_NAME, CacheStatistics.Stat.CACHE_HITS_FS.toString(), CacheStatistics.getFSHits());
			reporter.incrCounter( CacheableData.CACHING_COUNTER_GROUP_NAME, CacheStatistics.Stat.CACHE_HITS_HDFS.toString(), CacheStatistics.getHDFSHits());
			reporter.incrCounter( CacheableData.CACHING_COUNTER_GROUP_NAME, CacheStatistics.Stat.CACHE_WRITES_FSBUFF.toString(), CacheStatistics.getFSBuffWrites());
			reporter.incrCounter( CacheableData.CACHING_COUNTER_GROUP_NAME, CacheStatistics.Stat.CACHE_WRITES_FS.toString(), CacheStatistics.getFSWrites());
			reporter.incrCounter( CacheableData.CACHING_COUNTER_GROUP_NAME, CacheStatistics.Stat.CACHE_WRITES_HDFS.toString(), CacheStatistics.getHDFSWrites());
			reporter.incrCounter( CacheableData.CACHING_COUNTER_GROUP_NAME, CacheStatistics.Stat.CACHE_TIME_ACQR.toString(), CacheStatistics.getAcquireRTime());
			reporter.incrCounter( CacheableData.CACHING_COUNTER_GROUP_NAME, CacheStatistics.Stat.CACHE_TIME_ACQM.toString(), CacheStatistics.getAcquireMTime());
			reporter.incrCounter( CacheableData.CACHING_COUNTER_GROUP_NAME, CacheStatistics.Stat.CACHE_TIME_RLS.toString(), CacheStatistics.getReleaseTime());
			reporter.incrCounter( CacheableData.CACHING_COUNTER_GROUP_NAME, CacheStatistics.Stat.CACHE_TIME_EXP.toString(), CacheStatistics.getExportTime());
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
		
		//always reset stats because counters per map task (for case of JVM reuse)
		if( DMLScript.STATISTICS && !InfrastructureAnalyzer.isLocalMode() )
		{
			CacheStatistics.reset();
			Statistics.resetHOPRecompileTime();
			Statistics.resetJITCompileTime();
		}
	}
	
	private void pinResultVariables()
	{
		for( String var : _resultVars )
		{
			Data dat = _ec.getVariable(var);
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
