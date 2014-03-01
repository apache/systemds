/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.io;

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
import org.apache.hadoop.util.StringUtils;

import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;


public class MultipleOutputCommitter extends FileOutputCommitter 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	// maintain the map of matrix index to its final output dir
	// private HashMap<Byte, String> outputmap=new HashMap<Byte, String>();
	private String[] outputs;

	public void setupJob(JobContext context) throws IOException {
		super.setupJob(context);
		// get output file directories and creat directories
		JobConf conf = context.getJobConf();
		String[] outputs = MRJobConfiguration.getOutputs(conf);
		for (String dir : outputs) {
			Path path = new Path(dir);
			FileSystem fileSys = path.getFileSystem(conf);
			if (!fileSys.mkdirs(path))
			{
				if (!fileSys.mkdirs(path))
					LOG.error("Mkdirs failed to create " + path.toString());
			}
		}
	}

	public void cleanupJob(JobContext context) throws IOException {
		JobConf conf = context.getJobConf();
		// do the clean up of temporary directory
		Path outputPath = FileOutputFormat.getOutputPath(conf);
		if (outputPath != null) {
			FileSystem fileSys = outputPath.getFileSystem(conf);
			context.getProgressible().progress();
			if (fileSys.exists(outputPath)) {
				fileSys.delete(outputPath, true);
			}
		}
	}

	public void commitTask(TaskAttemptContext context) throws IOException {
		JobConf conf = context.getJobConf();

		// get the mapping between index to output filename
		outputs = MRJobConfiguration.getOutputs(conf);
		// byte[] indexes=MRJobConfiguration.getResultIndexes(conf);
		// for(int i=0; i<indexes.length; i++)
		// outputmap.put(indexes[i], outputs[i]);

		// LOG.info("outputmap has # entries "+outputmap.size());
		// for(Entry<Byte, String> e: outputmap.entrySet())
		// LOG.info(e.getKey()+" <--> "+e.getValue());

		Path taskOutputPath = getTempTaskOutputPath(context);
		TaskAttemptID attemptId = context.getTaskAttemptID();
		if (taskOutputPath != null) {
			FileSystem fs = taskOutputPath.getFileSystem(conf);
			context.getProgressible().progress();
			if (fs.exists(taskOutputPath)) {
				// Move the task outputs to their final places
				moveFinalTaskOutputs(context, fs, taskOutputPath);
				// Delete the temporary task-specific output directory
				if (!fs.delete(taskOutputPath, true)) {
					LOG.debug("Failed to delete the temporary output" + " directory of task: " + attemptId + " - "
							+ taskOutputPath);
				}
			}
		}
	}

	private void moveFinalTaskOutputs(TaskAttemptContext context, FileSystem fs, Path taskOutput) throws IOException {
		TaskAttemptID attemptId = context.getTaskAttemptID();
		context.getProgressible().progress();

		if (fs.getFileStatus(taskOutput).isDir()) {
			FileStatus[] files = fs.listStatus(taskOutput);
			if (files != null) {
				for (FileStatus file : files) {
					// only copy actual files
					if (file.isDir())
						continue;
					moveFileToDestination(context, fs, file.getPath());
				}
			}
		}
	}

	private void moveFileToDestination(TaskAttemptContext context, FileSystem fs, Path file) throws IOException {
		TaskAttemptID attemptId = context.getTaskAttemptID();
		Path finalPath = getFinalDestination(file, attemptId);
		// LOG.info("********** moving "+file+" to "+finalPath);
	//	System.out.println("********** moving "+file+" to "+finalPath);
		if (!fs.rename(file, finalPath)) {
			if (!fs.delete(finalPath, true)) {
				throw new IOException("Failed to delete earlier output " + finalPath + " so that can be rename with "
						+ file + " in task " + attemptId);
			}
			if (!fs.rename(file, finalPath)) {
				throw new IOException("Failed to save output " + finalPath + " so that can be rename with " + file
						+ " in task: " + attemptId);
			}
		}
		LOG.debug("Moved " + file + " to " + finalPath);
	}

	private Path getFinalDestination(Path file, TaskAttemptID attemptId) {
		int index = getOutputIndex(file);
		// LOG.info("filename: "+file+", index: "+index);
		return new Path(outputs[index], file.getName());
	}

	// XXXbhargav -- modified to check mapper outputs also
	// XXXbhargav -- Please check this yuanyuan.
	private int getOutputIndex(Path file) {
		String name = file.getName();
		int i = name.indexOf("-r-");
		if (i < 0)
			i = name.indexOf("-m-");

		if (i > 0)
			return Integer.parseInt(name.substring(0, i));
		else
			return 0;
	}

	Path getTempTaskOutputPath(TaskAttemptContext taskContext) {
		JobConf conf = taskContext.getJobConf();
		Path outputPath = FileOutputFormat.getOutputPath(conf);
		if (outputPath != null) {
			Path p = new Path(outputPath, (FileOutputCommitter.TEMP_DIR_NAME + Path.SEPARATOR + "_" + taskContext
					.getTaskAttemptID().toString()));
			try {
				FileSystem fs = p.getFileSystem(conf);
				return p.makeQualified(fs);
			} catch (IOException ie) {
				LOG.warn(StringUtils.stringifyException(ie));
				return p;
			}
		}
		return null;
	}
}
