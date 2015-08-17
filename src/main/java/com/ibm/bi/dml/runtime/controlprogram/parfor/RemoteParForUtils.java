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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import scala.Tuple2;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheStatistics;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheableData;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Stat;
import com.ibm.bi.dml.runtime.instructions.cp.Data;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;
import com.ibm.bi.dml.utils.Statistics;

/**
 * Common functionalities for parfor workers in MR jobs. Used by worker wrappers in
 * mappers (base RemoteParFor) and reducers (fused data partitioning and parfor)
 * 
 */
public class RemoteParForUtils 
{
	
	/**
	 * 
	 * @param reporter
	 * @param deltaTasks
	 * @param deltaIterations
	 */
	public static void incrementParForMRCounters(Reporter reporter, long deltaTasks, long deltaIterations)
	{
		//report parfor counters
		if( deltaTasks>0 )
			reporter.incrCounter(ParForProgramBlock.PARFOR_COUNTER_GROUP_NAME, Stat.PARFOR_NUMTASKS.toString(), deltaTasks);
		if( deltaIterations>0 )
			reporter.incrCounter(ParForProgramBlock.PARFOR_COUNTER_GROUP_NAME, Stat.PARFOR_NUMITERS.toString(), deltaIterations);
		
		if( DMLScript.STATISTICS  && !InfrastructureAnalyzer.isLocalMode() ) 
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
	
	/**
	 * 
	 * @param workerID
	 * @param vars
	 * @param resultVars
	 * @param out
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
	public static void exportResultVariables( long workerID, LocalVariableMap vars, ArrayList<String> resultVars, OutputCollector<Writable, Writable> out ) 
			throws DMLRuntimeException, IOException
	{
		exportResultVariables(workerID, vars, resultVars, null, out);
	}	
	
	/**
	 * For remote MR parfor workers.
	 * 
	 * @param workerID
	 * @param vars
	 * @param resultVars
	 * @param rvarFnames
	 * @param out
	 * @throws DMLRuntimeException
	 * @throws IOException
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
	 * @param workerID
	 * @param vars
	 * @param resultVars
	 * @param rvarFnames
	 * @throws DMLRuntimeException
	 * @throws IOException
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
	 * Cleanup all temporary files created by this SystemML process
	 * instance.
	 * 
	 */
	public static void cleanupWorkingDirectories()
	{
		if( !InfrastructureAnalyzer.isLocalMode() )
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
	 * 
	 * @param out
	 * @return
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
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
}
