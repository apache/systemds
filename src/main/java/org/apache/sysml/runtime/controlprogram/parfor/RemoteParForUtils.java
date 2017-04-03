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

package org.apache.sysml.runtime.controlprogram.parfor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import scala.Tuple2;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysml.runtime.controlprogram.caching.CacheStatistics;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.controlprogram.parfor.stat.Stat;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.util.LocalFileUtils;
import org.apache.sysml.utils.Statistics;

/**
 * Common functionalities for parfor workers in MR jobs. Used by worker wrappers in
 * mappers (base RemoteParFor) and reducers (fused data partitioning and parfor)
 * 
 */
public class RemoteParForUtils 
{

	public static void incrementParForMRCounters(Reporter reporter, long deltaTasks, long deltaIterations)
	{
		//report parfor counters
		if( deltaTasks>0 )
			reporter.incrCounter(ParForProgramBlock.PARFOR_COUNTER_GROUP_NAME, Stat.PARFOR_NUMTASKS.toString(), deltaTasks);
		if( deltaIterations>0 )
			reporter.incrCounter(ParForProgramBlock.PARFOR_COUNTER_GROUP_NAME, Stat.PARFOR_NUMITERS.toString(), deltaIterations);
		
		JobConf job = ConfigurationManager.getCachedJobConf();
		if( DMLScript.STATISTICS  && !InfrastructureAnalyzer.isLocalMode(job) ) 
		{
			//report cache statistics
			reporter.incrCounter( ParForProgramBlock.PARFOR_COUNTER_GROUP_NAME, Stat.PARFOR_JITCOMPILE.toString(), Statistics.getJITCompileTime());
			reporter.incrCounter( ParForProgramBlock.PARFOR_COUNTER_GROUP_NAME, Stat.PARFOR_JVMGC_COUNT.toString(), Statistics.getJVMgcCount());
			reporter.incrCounter( ParForProgramBlock.PARFOR_COUNTER_GROUP_NAME, Stat.PARFOR_JVMGC_TIME.toString(), Statistics.getJVMgcTime());
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
		
			//reset cache statistics to prevent overlapping reporting
			CacheStatistics.reset();
		}
	}

	public static void exportResultVariables( long workerID, LocalVariableMap vars, ArrayList<String> resultVars, OutputCollector<Writable, Writable> out ) 
			throws DMLRuntimeException, IOException
	{
		exportResultVariables(workerID, vars, resultVars, null, out);
	}	
	
	/**
	 * For remote MR parfor workers.
	 * 
	 * @param workerID worker id
	 * @param vars local variable map
	 * @param resultVars list of result variables
	 * @param rvarFnames ?
	 * @param out output collector
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 * @throws IOException if IOException occurs
	 */
	public static void exportResultVariables( long workerID, LocalVariableMap vars, ArrayList<String> resultVars, 
			                                  HashMap<String,String> rvarFnames, OutputCollector<Writable, Writable> out ) 
		throws DMLRuntimeException, IOException
	{
		//create key and value for reuse
		LongWritable okey = new LongWritable( workerID ); 
		Text ovalue = new Text();
		
		//foreach result variables probe if export necessary
		for( String rvar : resultVars )
		{
			Data dat = vars.get( rvar );
			
			//export output variable to HDFS (see RunMRJobs)
			if ( dat != null && dat.getDataType() == DataType.MATRIX ) 
			{
				MatrixObject mo = (MatrixObject) dat;
				if( mo.isDirty() )
				{
					if( ParForProgramBlock.ALLOW_REUSE_MR_PAR_WORKER && rvarFnames!=null )
					{
						String fname = rvarFnames.get( rvar );
						if( fname!=null )
							mo.setFileName( fname );
							
						//export result var (iff actually modified in parfor)
						mo.exportData(); //note: this is equivalent to doing it in close (currently not required because 1 Task=1Map tasks, hence only one map invocation)		
						rvarFnames.put(rvar, mo.getFileName());	
					}
					else
					{
						//export result var (iff actually modified in parfor)
						mo.exportData(); //note: this is equivalent to doing it in close (currently not required because 1 Task=1Map tasks, hence only one map invocation)
					}
					
					//pass output vars (scalars by value, matrix by ref) to result
					//(only if actually exported, hence in check for dirty, otherwise potential problems in result merge)
					String datStr = ProgramConverter.serializeDataObject(rvar, mo);
					ovalue.set( datStr );
					out.collect( okey, ovalue );
				}
			}	
		}
	}
	
	/**
	 * For remote Spark parfor workers. This is a simplified version compared to MR.
	 * 
	 * @param workerID worker id
	 * @param vars local variable map
	 * @param resultVars list of result variables
	 * @return list of result variables
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 * @throws IOException if IOException occurs
	 */
	public static ArrayList<String> exportResultVariables( long workerID, LocalVariableMap vars, ArrayList<String> resultVars) 
		throws DMLRuntimeException, IOException
	{
		ArrayList<String> ret = new ArrayList<String>();
		
		//foreach result variables probe if export necessary
		for( String rvar : resultVars )
		{
			Data dat = vars.get( rvar );
			
			//export output variable to HDFS (see RunMRJobs)
			if ( dat != null && dat.getDataType() == DataType.MATRIX ) 
			{
				MatrixObject mo = (MatrixObject) dat;
				if( mo.isDirty() )
				{
					//export result var (iff actually modified in parfor)
					mo.exportData(); 
					
					
					//pass output vars (scalars by value, matrix by ref) to result
					//(only if actually exported, hence in check for dirty, otherwise potential problems in result merge)
					ret.add( ProgramConverter.serializeDataObject(rvar, mo) );
				}
			}	
		}
		
		return ret;
	}
	
	/**
	 * Cleanup all temporary files created by this SystemML process.
	 */
	public static void cleanupWorkingDirectories()
	{
		//use the given job configuration for infrastructure analysis (see configure);
		//this is important for robustness w/ misconfigured classpath which also contains
		//core-default.xml and hence hides the actual cluster configuration; otherwise
		//there is missing cleanup of working directories 
		JobConf job = ConfigurationManager.getCachedJobConf();
		
		if( !InfrastructureAnalyzer.isLocalMode(job) )
		{
			//delete cache files
			CacheableData.cleanupCacheDir();
			//disable caching (prevent dynamic eviction)
			CacheableData.disableCaching();
			//cleanup working dir (e.g., of CP_FILE instructions)
			LocalFileUtils.cleanupWorkingDirectory();
		}
	}

	/**
	 * Cleanup all temporary files created by this SystemML process,
	 * on shutdown via exit or interrupt.
	 */
	public static void cleanupWorkingDirectoriesOnShutdown() {
		Runtime.getRuntime().addShutdownHook(
				new DeleteWorkingDirectoriesTask());
	}
	
	public static LocalVariableMap[] getResults( List<Tuple2<Long,String>> out, Log LOG ) 
		throws DMLRuntimeException
	{
		HashMap<Long,LocalVariableMap> tmp = new HashMap<Long,LocalVariableMap>();

		int countAll = 0;
		for( Tuple2<Long,String> entry : out )
		{
			Long key = entry._1();
			String val = entry._2();
			if( !tmp.containsKey( key ) )
        		tmp.put(key, new LocalVariableMap ());	   
			Object[] dat = ProgramConverter.parseDataObject( val );
        	tmp.get(key).put((String)dat[0], (Data)dat[1]);
        	countAll++;
		}

		if( LOG != null ) {
			LOG.debug("Num remote worker results (before deduplication): "+countAll);
			LOG.debug("Num remote worker results: "+tmp.size());
		}
		
		//create return array
		return tmp.values().toArray(new LocalVariableMap[0]);	
	}
	
	/**
	 * Task to be registered as shutdown hook in order to delete the 
	 * all working directories, including any remaining files, which 
	 * might not have been created  at time of registration.
	 */
	private static class DeleteWorkingDirectoriesTask extends Thread {
		@Override
		public void run() {
			cleanupWorkingDirectories();
		}
	}
}
