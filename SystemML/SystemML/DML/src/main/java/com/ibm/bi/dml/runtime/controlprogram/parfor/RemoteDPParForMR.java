/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.IOException;
import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.Counters.Group;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheStatistics;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheableData;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Stat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.PairWritableBlock;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.PairWritableCell;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.utils.Statistics;
import com.ibm.bi.dml.yarn.DMLAppMasterUtils;

/**
 * MR job class for submitting parfor remote MR jobs, controlling its execution and obtaining results.
 * 
 *
 */
public class RemoteDPParForMR
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected static final Log LOG = LogFactory.getLog(RemoteDPParForMR.class.getName());
	
	/**
	 * 
	 * @param pfid
	 * @param program
	 * @param taskFile
	 * @param resultFile
	 * @param enableCPCaching 
	 * @param mode
	 * @param numMappers
	 * @param replication
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static RemoteParForJobReturn runJob(long pfid, String itervar, String program, String resultFile, MatrixObject input, PDataPartitionFormat dpf, OutputInfo oi, boolean tSparseCol,
			                                   boolean enableCPCaching, ExecMode mode, int numReducers, int replication, int max_retry)  //opt params
		throws DMLRuntimeException
	{
		RemoteParForJobReturn ret = null;
		String jobname = "ParFor-DPEMR";
		long t0 = System.nanoTime();
		
		JobConf job;
		job = new JobConf( RemoteDPParForMR.class );
		job.setJobName(jobname+pfid);
		
		//maintain dml script counters
		Statistics.incrementNoOfCompiledMRJobs();
	
		try
		{
			/////
			//configure the MR job
		
			//set arbitrary CP program blocks that will perform in the reducers
			MRJobConfiguration.setProgramBlocks(job, program); 

			//enable/disable caching
			MRJobConfiguration.setParforCachingConfig(job, enableCPCaching);
		
			//setup input matrix
			Path path = new Path( input.getFileName() );
			long rlen = input.getNumRows();
			long clen = input.getNumColumns();
			int brlen = (int) input.getNumRowsPerBlock();
			int bclen = (int) input.getNumColumnsPerBlock();
			MRJobConfiguration.setPartitioningInfo(job, rlen, clen, brlen, bclen, InputInfo.BinaryBlockInputInfo, oi, dpf, 1, input.getFileName(), itervar, tSparseCol);
			job.setInputFormat(InputInfo.BinaryBlockInputInfo.inputFormatClass);
			FileInputFormat.setInputPaths(job, path);
			
			//set mapper and reducers classes
			job.setMapperClass(DataPartitionerRemoteMapper.class); 
			job.setReducerClass(RemoteDPParWorkerReducer.class); 
			
		    //set output format
		    job.setOutputFormat(SequenceFileOutputFormat.class);
		    
		    //set output path
		    MapReduceTool.deleteFileIfExistOnHDFS(resultFile);
		    FileOutputFormat.setOutputPath(job, new Path(resultFile));
		    
			//set the output key, value schema
		    
		    //parfor partitioning outputs (intermediates)
		    job.setMapOutputKeyClass(LongWritable.class);
		    if( oi == OutputInfo.BinaryBlockOutputInfo )
		    	job.setMapOutputValueClass(PairWritableBlock.class); 
		    else if( oi == OutputInfo.BinaryCellOutputInfo )
		    	job.setMapOutputValueClass(PairWritableCell.class);
		    else 
		    	throw new DMLRuntimeException("Unsupported intermrediate output info: "+oi);
		    //parfor exec output
		    job.setOutputKeyClass(LongWritable.class);
			job.setOutputValueClass(Text.class);
			
			//////
			//set optimization parameters

			//set the number of mappers and reducers 
			job.setNumReduceTasks( numReducers );			
			
			//disable automatic tasks timeouts and speculative task exec
			job.setInt("mapred.task.timeout", 0);			
			job.setMapSpeculativeExecution(false);
			
			//set up preferred custom serialization framework for binary block format
			if( MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION )
				MRJobConfiguration.addBinaryBlockSerializationFramework( job );
	
			//set up map/reduce memory configurations (if in AM context)
			DMLConfig config = ConfigurationManager.getConfig();
			DMLAppMasterUtils.setupMRJobRemoteMaxMemory(job, config);
			
			//disable JVM reuse
			job.setNumTasksToExecutePerJvm( 1 ); //-1 for unlimited 
			
			//set the replication factor for the results
			job.setInt("dfs.replication", replication);
			
			//set the max number of retries per map task
			//note: currently disabled to use cluster config
			//job.setInt("mapreduce.map.maxattempts", max_retry);
			
			// By default, the job executes in "cluster" mode.
			// Determine if we can optimize and run it in "local" mode.
			if ( mode == ExecMode.LOCAL ) {
				job.set("mapred.job.tracker", "local");	
				MRJobConfiguration.setStagingDir( job );
			}
			
			//set unique working dir
			MRJobConfiguration.setUniqueWorkingDir(job, mode);
			
			/////
			// execute the MR job			
			RunningJob runjob = JobClient.runJob(job);
			
			// Process different counters 
			Statistics.incrementNoOfExecutedMRJobs();
			Group pgroup = runjob.getCounters().getGroup(ParForProgramBlock.PARFOR_COUNTER_GROUP_NAME);
			int numTasks = (int)pgroup.getCounter( Stat.PARFOR_NUMTASKS.toString() );
			int numIters = (int)pgroup.getCounter( Stat.PARFOR_NUMITERS.toString() );
			if( DMLScript.STATISTICS && !InfrastructureAnalyzer.isLocalMode() ) {
				Statistics.incrementJITCompileTime( pgroup.getCounter( Stat.PARFOR_JITCOMPILE.toString() ) );
				Statistics.incrementJVMgcCount( pgroup.getCounter( Stat.PARFOR_JVMGC_COUNT.toString() ) );
				Statistics.incrementJVMgcTime( pgroup.getCounter( Stat.PARFOR_JVMGC_TIME.toString() ) );
				Group cgroup = runjob.getCounters().getGroup(CacheableData.CACHING_COUNTER_GROUP_NAME.toString());
				CacheStatistics.incrementMemHits((int)cgroup.getCounter( CacheStatistics.Stat.CACHE_HITS_MEM.toString() ));
				CacheStatistics.incrementFSBuffHits((int)cgroup.getCounter( CacheStatistics.Stat.CACHE_HITS_FSBUFF.toString() ));
				CacheStatistics.incrementFSHits((int)cgroup.getCounter( CacheStatistics.Stat.CACHE_HITS_FS.toString() ));
				CacheStatistics.incrementHDFSHits((int)cgroup.getCounter( CacheStatistics.Stat.CACHE_HITS_HDFS.toString() ));
				CacheStatistics.incrementFSBuffWrites((int)cgroup.getCounter( CacheStatistics.Stat.CACHE_WRITES_FSBUFF.toString() ));
				CacheStatistics.incrementFSWrites((int)cgroup.getCounter( CacheStatistics.Stat.CACHE_WRITES_FS.toString() ));
				CacheStatistics.incrementHDFSWrites((int)cgroup.getCounter( CacheStatistics.Stat.CACHE_WRITES_HDFS.toString() ));
				CacheStatistics.incrementAcquireRTime(cgroup.getCounter( CacheStatistics.Stat.CACHE_TIME_ACQR.toString() ));
				CacheStatistics.incrementAcquireMTime(cgroup.getCounter( CacheStatistics.Stat.CACHE_TIME_ACQM.toString() ));
				CacheStatistics.incrementReleaseTime(cgroup.getCounter( CacheStatistics.Stat.CACHE_TIME_RLS.toString() ));
				CacheStatistics.incrementExportTime(cgroup.getCounter( CacheStatistics.Stat.CACHE_TIME_EXP.toString() ));
			}
				
			// read all files of result variables and prepare for return
			LocalVariableMap[] results = readResultFile(job, resultFile); 

			ret = new RemoteParForJobReturn(runjob.isSuccessful(), 
					                        numTasks, numIters, 
					                        results);  	
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		finally
		{
			// remove created files 
			try
			{
				MapReduceTool.deleteFileIfExistOnHDFS(new Path(resultFile), job);
			}
			catch(IOException ex)
			{
				throw new DMLRuntimeException(ex);
			}
		}
		
		if( DMLScript.STATISTICS ){
			long t1 = System.nanoTime();
			Statistics.maintainCPHeavyHitters("MR-Job_"+jobname, t1-t0);
		}
		
		return ret;
	}
	

	/**
	 * Result file contains hierarchy of workerID-resultvar(incl filename). We deduplicate
	 * on the workerID. Without JVM reuse each task refers to a unique workerID, so we
	 * will not find any duplicates. With JVM reuse, however, each slot refers to a workerID, 
	 * and there are duplicate filenames due to partial aggregation and overwrite of fname 
	 * (the RemoteParWorkerMapper ensures uniqueness of those files independent of the 
	 * runtime implementation). 
	 * 
	 * @param job 
	 * @param fname
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static LocalVariableMap [] readResultFile( JobConf job, String fname )
		throws DMLRuntimeException, IOException
	{
		HashMap<Long,LocalVariableMap> tmp = new HashMap<Long,LocalVariableMap>();

		FileSystem fs = FileSystem.get(job);
		Path path = new Path(fname);
		LongWritable key = new LongWritable(); //workerID
		Text value = new Text();               //serialized var header (incl filename)
		
		int countAll = 0;
		for( Path lpath : DataConverter.getSequenceFilePaths(fs, path) )
		{
			SequenceFile.Reader reader = new SequenceFile.Reader(FileSystem.get(job),lpath,job);
			try
			{
				while( reader.next(key, value) )
				{
					//System.out.println("key="+key.get()+", value="+value.toString());
					if( !tmp.containsKey( key.get() ) )
		        		tmp.put(key.get(), new LocalVariableMap ());	   
					Object[] dat = ProgramConverter.parseDataObject( value.toString() );
		        	tmp.get( key.get() ).put((String)dat[0], (Data)dat[1]);
		        	countAll++;
				}
			}	
			finally
			{
				if( reader != null )
					reader.close();
			}
		}		

		LOG.debug("Num remote worker results (before deduplication): "+countAll);
		LOG.debug("Num remote worker results: "+tmp.size());

		//create return array
		return tmp.values().toArray(new LocalVariableMap[0]);	
	}
}
