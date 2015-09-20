/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
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
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheStatistics;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheableData;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.StatisticMonitor;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDHandler;
import com.ibm.bi.dml.runtime.instructions.cp.Data;
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
	
	//cache for future reuse (in case of JVM reuse)
	//NOTE: Hashmap to support multiple parfor MR jobs for local mode and if JVM reuse across jobs
	private static HashMap<String,RemoteParWorkerMapper> _sCache = null; 
	
	//MR ParWorker attributes  
	protected String  _stringID       = null; 
	protected HashMap<String, String> _rvarFnames = null; 
	
	static
	{
		//init cache (once per JVM)
		_sCache = new HashMap<String, RemoteParWorkerMapper>();
	}
	
	
	public RemoteParWorkerMapper( ) 
	{
		//only used if JVM reuse is enabled in order to ensure consistent output 
		//filenames across tasks of one reused worker (preaggregation)
		_rvarFnames = new HashMap<String, String>();
	}
	
	/**
	 * 
	 */
	@Override
	public void map(LongWritable key, Text value, OutputCollector<Writable, Writable> out, Reporter reporter) 
		throws IOException
	{
		LOG.trace("execute RemoteParWorkerMapper "+_stringID+" ("+_workerID+")");
		
		//state for jvm reuse and multiple iterations 
		long numIters = getExecutedIterations(); 
		
		try 
		{
			//parse input task
			Task lTask = Task.parseCompactString( value.toString() );
			
			//execute task (on error: re-try via Hadoop)
			executeTask( lTask );
		
			//write output if required (matrix indexed write)
			RemoteParForUtils.exportResultVariables( _workerID, _ec.getVariables(), _resultVars, _rvarFnames, out );
		}
		catch(Exception ex)
		{
			//throw IO exception to adhere to API specification
			throw new IOException("ParFOR: Failed to execute task.",ex);
		}
		
		//statistic maintenance
		RemoteParForUtils.incrementParForMRCounters(reporter, 1, getExecutedIterations()-numIters);
		
		//print heaver hitter per task
		JobConf job = ConfigurationManager.getCachedJobConf();
		if( DMLScript.STATISTICS && !InfrastructureAnalyzer.isLocalMode(job) )
			LOG.info("\nSystemML Statistics:\nHeavy hitter instructions (name, time, count):\n" + Statistics.getHeavyHitters(10));	
	}

	/**
	 * 
	 */
	@Override
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
										
					_rvarFnames     = tmp._rvarFnames;
					
					requiresConfigure = false;
				}
			}
		}
		
		if( requiresConfigure )
		{
			LOG.trace("configure RemoteParWorkerMapper "+job.get("mapred.tip.id"));
			
			try
			{
				_stringID = job.get("mapred.tip.id"); //task ID
				_workerID = IDHandler.extractIntID(_stringID); //int task ID

				//use the given job configuration as source for all new job confs 
				//NOTE: this is required because on HDP 2.3, the classpath of mr tasks contained hadoop-common.jar 
				//which includes a core-default.xml configuration which hides the actual default cluster configuration
				//in the context of mr jobs (for example this config points to local fs instead of hdfs by default). 
				if( !InfrastructureAnalyzer.isLocalMode(job) ) {
					ConfigurationManager.setCachedJobConf(job);
				}
				
				//create local runtime program
				String in = MRJobConfiguration.getProgramBlocks(job);
				ParForBody body = ProgramConverter.parseParForBody(in, (int)_workerID);
				_childBlocks = body.getChildBlocks();
				_ec          = body.getEc();				
				_resultVars  = body.getResultVarNames();
		
				//init local cache manager 
				if( !CacheableData.isCachingActive() ) {
					String uuid = IDHandler.createDistributedUniqueID();
					LocalFileUtils.createWorkingDirectoryWithUUID( uuid );
					CacheableData.initCaching( uuid ); //incl activation, cache dir creation (each map task gets its own dir for simplified cleanup)
				}				
				if( !CacheableData.cacheEvictionLocalFilePrefix.contains("_") ){ //account for local mode
					CacheableData.cacheEvictionLocalFilePrefix = CacheableData.cacheEvictionLocalFilePrefix +"_" + _workerID; 
				}
				
				//ensure that resultvar files are not removed
				super.pinResultVariables();
				
				//enable/disable caching (if required)
				boolean cpCaching = MRJobConfiguration.getParforCachingConfig( job );
				if( !cpCaching )
					CacheableData.disableCaching();
		
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
		if( DMLScript.STATISTICS && !InfrastructureAnalyzer.isLocalMode(job) )
		{
			CacheStatistics.reset();
			Statistics.reset();
		}
	}

	/**
	 * 
	 */
	@Override
	public void close() 
		throws IOException 
	{
		//cleanup cache and local tmp dir
		RemoteParForUtils.cleanupWorkingDirectories();
		
		//change cache status for jvm_reuse (make empty allows us to
		//reuse in-memory objects if still present, re-load from HDFS
		//if evicted by garbage collector - without this approach, we
		//could not cleanup the local working dir, because this would 
		//delete evicted matrices as well. 
		if( ParForProgramBlock.ALLOW_REUSE_MR_PAR_WORKER )
		{
			for( RemoteParWorkerMapper pw : _sCache.values() )
			{
				LocalVariableMap vars = pw._ec.getVariables();
				for( String varName : vars.keySet() )
				{
					Data dat = vars.get(varName);
					if( dat instanceof MatrixObject )
						((MatrixObject)dat).setEmptyStatus();
				}
			}
		}
		
		//ensure caching is not disabled for CP in local mode
		CacheableData.enableCaching();
	}

}
