/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
}
