/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.lib.NLineInputFormat;
import org.apache.hadoop.mapred.lib.NullOutputFormat;

import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;


public class CleanupMR 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final Log LOG = LogFactory.getLog(CleanupMR.class.getName());
	
	public static boolean runJob( DMLConfig conf ) 
		throws Exception
	{
		boolean ret = false;
		
		try
		{
			JobConf job;
			job = new JobConf(CleanupMR.class);
			job.setJobName("Cleanup-MR");
			
			//set up SystemML local tmp dir
			String dir = conf.getTextValue(DMLConfig.LOCAL_TMP_DIR);
			MRJobConfiguration.setSystemMLLocalTmpDir(job, dir); 
			
			//set mappers, reducers 
			int numNodes = InfrastructureAnalyzer.getRemoteParallelNodes();
			job.setMapperClass(CleanupMapper.class); //map-only
			job.setNumMapTasks(numNodes); //numMappers
			job.setNumReduceTasks( 0 );			
			
			//set input/output format, input path
			String inFileName = conf.getTextValue(DMLConfig.SCRATCH_SPACE)+"/cleanup_tasks";
			job.setInputFormat(NLineInputFormat.class);
		    job.setOutputFormat(NullOutputFormat.class);
		    
			Path path = new Path( inFileName );
		    FileInputFormat.setInputPaths(job, path);
		    writeCleanupTasksToFile(path, numNodes);
		    
			//disable automatic tasks timeouts and speculative task exec
			job.setInt("mapred.task.timeout", 0);			
			job.setMapSpeculativeExecution(false);
			
			/////
			// execute the MR job			
			RunningJob runjob = JobClient.runJob(job);
			
			ret = runjob.isSuccessful();
		}
		catch(Exception ex)
		{
			//don't raise an exception, just gracefully an error message.
			LOG.error("Failed to run cleanup MR job. ",ex);
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param path
	 * @param numTasks
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
	private static void writeCleanupTasksToFile(Path path, int numTasks)
		throws DMLRuntimeException, IOException
	{
		BufferedWriter br = null;
		try
		{
			FileSystem fs = FileSystem.get(ConfigurationManager.getCachedJobConf());
			br = new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));
	        
			for( int i=1; i<=numTasks; i++ )
				br.write( String.valueOf("CLEANUP TASK "+i)+"\n" );
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Error writing cleanup tasks to taskfile "+path.toString(), ex);
		}
		finally
		{
			if( br != null )
				br.close();
		}
	}
	
	public static class CleanupMapper implements Mapper<LongWritable, Text, Writable, Writable>
	{
		private static final Log LOG = LogFactory.getLog(CleanupMapper.class.getName());
		
		//file name local tmp dir  
		protected String _dir = null; 
		
		public CleanupMapper( ) 
		{
			
		}
		
		@Override
		public void map(LongWritable key, Text value, OutputCollector<Writable, Writable> out, Reporter reporter) 
			throws IOException
		{
			try 
			{
				String task = value.toString();
				LOG.info("Running cleanup task: "+task+" ("+_dir+") ... ");
				
				int count = LocalFileUtils.cleanupRcWorkingDirectory(_dir);
				LOG.info("Done - deleted "+count+" files.");
			}
			catch(Exception ex)
			{
				//throw IO exception to adhere to API specification
				throw new IOException("Failed to execute cleanup task.",ex);
			}
		}
	
		@Override
		public void configure(JobConf job)
		{
			_dir = MRJobConfiguration.getSystemMLLocalTmpDir(job);
		}
		
		@Override
		public void close()
		{
			
		}
	}
}
