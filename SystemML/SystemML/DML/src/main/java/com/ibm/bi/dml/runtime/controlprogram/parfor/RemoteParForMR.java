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
import org.apache.hadoop.mapred.lib.NLineInputFormat;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheStatistics;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheableData;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Stat;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
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
public class RemoteParForMR
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected static final Log LOG = LogFactory.getLog(RemoteParForMR.class.getName());
	
	/**
	 * 
	 * @param pfid
	 * @param program
	 * @param taskFile
	 * @param resultFile
	 * @param _enableCPCaching 
	 * @param mode
	 * @param numMappers
	 * @param replication
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static RemoteParForJobReturn runJob(long pfid, String program, String taskFile, String resultFile, MatrixObject colocatedDPMatrixObj, //inputs
			                                   boolean enableCPCaching, ExecMode mode, int numMappers, int replication, int max_retry, long minMem, boolean jvmReuse)  //opt params
		throws DMLRuntimeException
	{
		RemoteParForJobReturn ret = null;
		String jobname = "ParFor-EMR";
		long t0 = System.nanoTime();
		
		JobConf job;
		job = new JobConf( RemoteParForMR.class );
		job.setJobName(jobname+pfid);
		
		//maintain dml script counters
		Statistics.incrementNoOfCompiledMRJobs();
	
		try
		{
			/////
			//configure the MR job
		
			//set arbitrary CP program blocks that will perform in the mapper
			MRJobConfiguration.setProgramBlocks(job, program); 
			
			//enable/disable caching
			MRJobConfiguration.setParforCachingConfig(job, enableCPCaching);
			
			//set mappers, reducers, combiners
			job.setMapperClass(RemoteParWorkerMapper.class); //map-only

			//set input format (one split per row, NLineInputFormat default N=1)
			if( ParForProgramBlock.ALLOW_DATA_COLOCATION && colocatedDPMatrixObj != null )
			{
				job.setInputFormat(RemoteParForColocatedNLineInputFormat.class);
				MRJobConfiguration.setPartitioningFormat(job, colocatedDPMatrixObj.getPartitionFormat());
				MatrixCharacteristics mc = ((MatrixFormatMetaData)colocatedDPMatrixObj.getMetaData()).getMatrixCharacteristics();
				MRJobConfiguration.setPartitioningBlockNumRows(job, mc.get_rows_per_block());
				MRJobConfiguration.setPartitioningBlockNumCols(job, mc.get_cols_per_block());
				MRJobConfiguration.setPartitioningFilename(job, colocatedDPMatrixObj.getFileName());
			}
			else //default case 
			{
				job.setInputFormat(NLineInputFormat.class);
			}
			
			//set the input path and output path 
		    FileInputFormat.setInputPaths(job, new Path(taskFile));
			
		    //set output format
		    job.setOutputFormat(SequenceFileOutputFormat.class);
		    
		    //set output path
		    MapReduceTool.deleteFileIfExistOnHDFS(resultFile);
		    FileOutputFormat.setOutputPath(job, new Path(resultFile));
		    
			//set the output key, value schema
			job.setMapOutputKeyClass(LongWritable.class);
			job.setMapOutputValueClass(Text.class);			
			job.setOutputKeyClass(LongWritable.class);
			job.setOutputValueClass(Text.class);
			
			//////
			//set optimization parameters

			//set the number of mappers and reducers 
			job.setNumMapTasks(numMappers); //numMappers
			job.setNumReduceTasks( 0 );			
			//job.setInt("mapred.map.tasks.maximum", 1); //system property
			//job.setInt("mapred.tasktracker.tasks.maximum",1); //system property
			//job.setInt("mapred.jobtracker.maxtasks.per.job",1); //system property

			//use FLEX scheduler configuration properties
			if( ParForProgramBlock.USE_FLEX_SCHEDULER_CONF )
			{
				job.setInt("flex.priority",0); //highest
				
				job.setInt("flex.map.min", 0);
				job.setInt("flex.map.max", numMappers);
				job.setInt("flex.reduce.min", 0);
				job.setInt("flex.reduce.max", numMappers);
			}
			
			//set jvm memory size (if require)
			String memKey = "mapred.child.java.opts";
			if( minMem > 0 && minMem > InfrastructureAnalyzer.extractMaxMemoryOpt(job.get(memKey)) )
			{
				InfrastructureAnalyzer.setMaxMemoryOpt(job, memKey, minMem);
				LOG.warn("Forcing '"+memKey+"' to -Xmx"+minMem/(1024*1024)+"M." );
			}
			
			//disable automatic tasks timeouts and speculative task exec
			job.setInt("mapred.task.timeout", 0);			
			job.setMapSpeculativeExecution(false);
			
			//set up map/reduce memory configurations (if in AM context)
			DMLConfig config = ConfigurationManager.getConfig();
			DMLAppMasterUtils.setupMRJobRemoteMaxMemory(job, config);
			
			//enables the reuse of JVMs (multiple tasks per MR task)
			if( jvmReuse )
				job.setNumTasksToExecutePerJvm(-1); //unlimited
			
			//set sort io buffer (reduce unnecessary large io buffer, guaranteed memory consumption)
			job.setInt("io.sort.mb", 8); //8MB
			
			//set the replication factor for the results
			job.setInt("dfs.replication", replication);
			
			//set the max number of retries per map task
			//  disabled job-level configuration to respect cluster configuration
			//  note: this refers to hadoop2, hence it never had effect on mr1
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
				MapReduceTool.deleteFileIfExistOnHDFS(new Path(taskFile), job);
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
