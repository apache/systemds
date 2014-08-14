/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.io.hadoopfix;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.lib.IdentityMapper;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.util.MapReduceTool;


public class Test extends Configured 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@SuppressWarnings("rawtypes")
	static class InnerMapper extends IdentityMapper
	{

		public void configure(JobConf conf)
		{ 
			String inputName=conf.get("map.input.file"); 
			//System.out.println(inputName);
		}
	
	}

	public static void runJob(String[] arg0) throws Exception 
	{ 	
		JobConf job = new JobConf(Test.class);
		
		MultipleInputs.addInputPath(job, new Path("A"), SequenceFileInputFormat.class); 
		MultipleInputs.addInputPath(job, new Path("B"), SequenceFileInputFormat.class); 
		
		job.setMapperClass(InnerMapper.class);
		job.setMapOutputKeyClass(MatrixIndexes.class);
		job.setMapOutputValueClass(MatrixBlock.class);
		job.setOutputFormat(SequenceFileOutputFormat.class);
		job.setOutputKeyClass(MatrixIndexes.class);
		job.setOutputValueClass(MatrixBlock.class);
	
		
		// configure reducers
		job.setNumReduceTasks(0);
		
		// configure output
		Path outputDir=new Path("temp");
		FileOutputFormat.setOutputPath(job, outputDir);
		MapReduceTool.deleteFileIfExistOnHDFS(outputDir, job);
		job.setInt("dfs.replication", 1);
		
		JobClient.runJob(job);
	}
	
	public static void main(String[] args) throws Exception {
		Test.runJob(args);
	}
}