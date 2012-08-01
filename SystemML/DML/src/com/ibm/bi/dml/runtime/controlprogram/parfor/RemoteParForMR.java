package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.IOException;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.Counters;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.lib.NLineInputFormat;

import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Stat;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.utils.DMLRuntimeException;

/**
 * MR job class for submitting parfor remote MR jobs, controlling its execution and obtaining results.
 * 
 *
 */
public class RemoteParForMR
{
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
	public static RemoteParForJobReturn runJob(long pfid, String program, String taskFile, String resultFile, //inputs
			                       ExecMode mode, int numMappers, int replication, int max_retry)  //opt params
		throws DMLRuntimeException
	{
		RemoteParForJobReturn ret = null;
		
		JobConf job;
		job = new JobConf( RemoteParForMR.class );
		job.setJobName("ParFOR-MR_"+pfid);
		
	
		try
		{
			/////
			//configure the MR job
		
			//set arbitrary CP program blocks that will perform in the mapper
			MRJobConfiguration.setProgramBlocksInMapper(job, program); 
			
			//set mappers, reducers, combiners
			job.setMapperClass(RemoteParWorkerMapper.class); //map-only

			//set input format (one split per row, NLineInputFormat default N=1)
			job.setInputFormat(NLineInputFormat.class);
			
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

			//use FLEX scheudler configuration properties
			//System.out.println("numMappers="+numMappers);
			if( ParForProgramBlock.USE_FLEX_SCHEDULER_CONF )
			{
				job.setInt("flex.map.min", 0);
				job.setInt("flex.map.max", numMappers);
				job.setInt("flex.reduce.min", 0);
				job.setInt("flex.reduce.max", numMappers);
			}
			
			//disable automatic tasks timeouts and speculative task exec
			job.setInt("mapred.task.timeout", 0);			
			job.setMapSpeculativeExecution(false);
			
			//enables the reuse of JVMs (multiple tasks per MR task)
			if( ParForProgramBlock.ALLOW_REUSE_MR_JVMS )
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
			Counters group = runjob.getCounters();
			int numTasks = (int)group.getCounter( Stat.PARFOR_NUMTASKS );
			int numIters = (int)group.getCounter( Stat.PARFOR_NUMITERS );
			
			
			// read all files of result variables and prepare for return
			LocalVariableMap [] results = readResultFile(resultFile); 

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
	public static LocalVariableMap [] readResultFile(String fname)
		throws DMLRuntimeException, IOException
	{
		LocalVariableMap [] ret = null;
		HashMap<Long,LocalVariableMap> tmp = new HashMap<Long,LocalVariableMap>();

		SequenceFile.Reader reader = null;
		
		Path path = new Path(fname);
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		
		//foreach file in path
		FileStatus[] status = fs.listStatus(path);
		for( FileStatus f : status )
		{
			if( f.getPath().getName().startsWith("_") ) //reject system-internal filenames
				continue;
				
			try
			{
				reader = new SequenceFile.Reader(fs, f.getPath(), conf);
				
				LongWritable key = new LongWritable();
		        Text value = new Text();
		        while( reader.next(key, value) ) 
		        {
		        	if( !tmp.containsKey( key.get() ) )
		        		tmp.put(key.get(), new LocalVariableMap ());	        	
		        	Object[] dat = ProgramConverter.parseDataObject( value.toString() );
		        	tmp.get( key.get() ).put((String)dat[0], (Data)dat[1]);
		        }
			}
			catch(Exception ex)
			{
				throw new DMLRuntimeException("Error reading results from resultfile "+fname, ex);
			}
			finally
			{
		        if(reader!=null)
		        	reader.close();				
			}
		}

		
		//create return
		ret = tmp.values().toArray(new LocalVariableMap[0]);	
			
		return ret;
	}
}
