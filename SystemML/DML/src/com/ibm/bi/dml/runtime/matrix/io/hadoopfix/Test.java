package com.ibm.bi.dml.runtime.matrix.io.hadoopfix;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.lib.IdentityMapper;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.util.MapReduceTool;


public class Test extends Configured {

	static class InnerMapper extends IdentityMapper
	{

		public void configure(JobConf conf)
		{ 
			String inputName=conf.get("map.input.file"); 
			System.out.println(inputName);
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