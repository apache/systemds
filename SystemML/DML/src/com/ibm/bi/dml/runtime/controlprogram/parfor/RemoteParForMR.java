package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.IOException;
import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.Counters;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.lib.NLineInputFormat;

import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
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
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.Statistics;

/**
 * MR job class for submitting parfor remote MR jobs, controlling its execution and obtaining results.
 * 
 *
 */
public class RemoteParForMR
{
	protected static final Log LOG = LogFactory.getLog(RemoteParForMR.class.getName());
	
	/**
	 * 
	 * @param pfid
	 * @param program
	 * @param taskFile
	 * @param resultFile
	 * @param mode
	 * @param numMappers
	 * @param replication
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static RemoteParForJobReturn runJob(long pfid, String program, String taskFile, String resultFile, MatrixObject colocatedDPMatrixObj, //inputs
			                                   ExecMode mode, int numMappers, int replication, int max_retry, long minMem, boolean jvmReuse)  //opt params
		throws DMLRuntimeException
	{
		RemoteParForJobReturn ret = null;
		
		JobConf job;
		job = new JobConf( RemoteParForMR.class );
		job.setJobName("ParFor_Execute-MR"+pfid);
		
		//maintain dml script counters
		Statistics.incrementNoOfCompiledMRJobs();
	
		try
		{
			/////
			//configure the MR job
		
			//set arbitrary CP program blocks that will perform in the mapper
			MRJobConfiguration.setProgramBlocksInMapper(job, program); 
			
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
			
			//enables the reuse of JVMs (multiple tasks per MR task)
			if( jvmReuse )
				job.setNumTasksToExecutePerJvm(-1); //unlimited
			
			//set the replication factor for the results
			job.setInt("dfs.replication", replication);
			
			//set the max number of retries per map task
			job.setInt("mapreduce.map.maxattempts", max_retry);
			
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
			Counters group = runjob.getCounters();
			int numTasks = (int)group.getCounter( Stat.PARFOR_NUMTASKS );
			int numIters = (int)group.getCounter( Stat.PARFOR_NUMITERS );
			if( CacheableData.CACHING_STATS && !InfrastructureAnalyzer.isLocalMode() )
			{
				CacheStatistics.incrementMemHits((int)group.getCounter( CacheStatistics.Stat.CACHE_HITS_MEM ));
				CacheStatistics.incrementFSHits((int)group.getCounter( CacheStatistics.Stat.CACHE_HITS_FS ));
				CacheStatistics.incrementHDFSHits((int)group.getCounter( CacheStatistics.Stat.CACHE_HITS_HDFS ));
				CacheStatistics.incrementFSWrites((int)group.getCounter( CacheStatistics.Stat.CACHE_WRITES_FS ));
				CacheStatistics.incrementHDFSWrites((int)group.getCounter( CacheStatistics.Stat.CACHE_WRITES_HDFS ));
			}
				
			// read all files of result variables and prepare for return
			LocalVariableMap[] results = readResultFile(resultFile); 

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
		
		return ret;
	}
	

	/**
	 * 
	 * @param fname
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static LocalVariableMap [] readResultFile( String fname )
		throws DMLRuntimeException, IOException
	{
		HashMap<Long,LocalVariableMap> tmp = new HashMap<Long,LocalVariableMap>();

		JobConf job = new JobConf();
		Path path = new Path(fname);
		FileInputFormat.addInputPath(job, path); 
		
		SequenceFileInputFormat<LongWritable,Text> informat = new SequenceFileInputFormat<LongWritable,Text>();
		InputSplit[] splits = informat.getSplits(job, 1);
		LongWritable key = new LongWritable();
		Text value = new Text();
		
		for(InputSplit split: splits)
		{
			RecordReader<LongWritable,Text> reader = informat.getRecordReader(split, job, Reporter.NULL);
			try
			{
				while( reader.next(key, value) )
				{
					if( !tmp.containsKey( key.get() ) )
		        		tmp.put(key.get(), new LocalVariableMap ());	   
					Object[] dat = ProgramConverter.parseDataObject( value.toString() );
		        	tmp.get( key.get() ).put((String)dat[0], (Data)dat[1]);
				}
			}	
			finally
			{
				if( reader != null )
					reader.close();
			}
		}		

		//create return array
		return tmp.values().toArray(new LocalVariableMap[0]);	
	}
}
