/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.data;

import java.io.IOException;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileOutputCommitter;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.JobContext;
import org.apache.hadoop.mapred.TaskAttemptContext;
import org.apache.hadoop.mapred.TaskAttemptID;

import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;


public class MultipleOutputCommitter extends FileOutputCommitter 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	// maintain the map of matrix index to its final output dir
	// private HashMap<Byte, String> outputmap=new HashMap<Byte, String>();
	private String[] outputs;

	@Override
	public void setupJob(JobContext context) 
		throws IOException 
	{
		super.setupJob(context);
		// get output file directories and create directories
		JobConf conf = context.getJobConf();
		String[] loutputs = MRJobConfiguration.getOutputs(conf);
		for (String dir : loutputs) {
			Path path = new Path(dir);
			FileSystem fs = path.getFileSystem(conf);
			if( !fs.mkdirs(path) )
				LOG.error("Mkdirs failed to create " + path.toString());
		}
	}

	@Override
	public void cleanupJob(JobContext context) 
		throws IOException 
	{
		JobConf conf = context.getJobConf();
		// do the clean up of temporary directory
		Path outputPath = FileOutputFormat.getOutputPath(conf);
		if (outputPath != null) {
			FileSystem fs = outputPath.getFileSystem(conf);
			context.getProgressible().progress();
			if( fs.exists(outputPath) ) 
				fs.delete(outputPath, true);
		}
	}

	@Override
	public void commitTask(TaskAttemptContext context) 
		throws IOException 
	{
		JobConf conf = context.getJobConf();
		TaskAttemptID attemptId = context.getTaskAttemptID();
		
		// get the mapping between index to output filename
		outputs = MRJobConfiguration.getOutputs(conf);
		
		//get temp task output path (compatible with hadoop1 and hadoop2)
		Path taskOutPath = FileOutputFormat.getWorkOutputPath(conf);
		FileSystem fs = taskOutPath.getFileSystem(conf);
		if( !fs.exists(taskOutPath) )
			throw new IOException("Task output path "+ taskOutPath.toString() + "does not exist.");
		
		// Move the task outputs to their final places
		context.getProgressible().progress();
		moveFinalTaskOutputs(context, fs, taskOutPath);
		
		// Delete the temporary task-specific output directory
		if( !fs.delete(taskOutPath, true) ) 
			LOG.debug("Failed to delete the temporary output directory of task: " + attemptId + " - " + taskOutPath);
	}
	
	/**
	 * 
	 * @param context
	 * @param fs
	 * @param taskOutput
	 * @throws IOException
	 */
	private void moveFinalTaskOutputs(TaskAttemptContext context, FileSystem fs, Path taskOutput)
		throws IOException 
	{
		context.getProgressible().progress();
		
		if( fs.getFileStatus(taskOutput).isDirectory() ) 
		{
			FileStatus[] files = fs.listStatus(taskOutput);
			if (files != null)
				for (FileStatus file : files) //for all files
					if( !file.isDirectory() ) //skip directories
						moveFileToDestination(context, fs, file.getPath());
		}
	}
	
	/**
	 * 
	 * @param context
	 * @param fs
	 * @param file
	 * @throws IOException
	 */
	private void moveFileToDestination(TaskAttemptContext context, FileSystem fs, Path file) 
		throws IOException 
	{
		JobConf conf = context.getJobConf();
		TaskAttemptID attemptId = context.getTaskAttemptID();
		
		//get output index and final destination
		String taskType = (conf.getBoolean(JobContext.TASK_ISMAP, true)) ? "m" : "r";
		String name =  file.getName(); 
		int charIx = name.indexOf("-"+taskType+"-");
		int index = Integer.parseInt(name.substring(0, charIx));
		Path finalPath = new Path(outputs[index], file.getName());
		
		//move file from 'file' to 'finalPath'
		if( !fs.rename(file, finalPath) ) 
		{
			if (!fs.delete(finalPath, true))
				throw new IOException("Failed to delete earlier output " + finalPath + " for rename of " + file + " in task " + attemptId);
			if (!fs.rename(file, finalPath)) 
				throw new IOException("Failed to save output " + finalPath + " for rename of " + file + " in task: " + attemptId);
		}
	}

}
