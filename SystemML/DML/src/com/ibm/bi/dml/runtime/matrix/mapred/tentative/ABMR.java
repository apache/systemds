/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred.tentative;

import java.util.Vector;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.TaggedTripleIndexes;
import com.ibm.bi.dml.runtime.util.MapReduceTool;


public class ABMR 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	public static JobConf runJob(String inputs, String aggBinaryOperation, String output, 
		int numReducers, int replication, long nr1, long nc1, long nr2, long nc2, int bnr1, int bnc1, int bnr2, int bnc2) 
	throws Exception
	{
	//	if(true)
	//		throw new RuntimeException("Has to fix the problem of starting from 1 instead of 0");
		
		JobConf job;
		job = new JobConf(ABMR.class);
		job.setJobName("AggregateBinary");
		
		job.set(ABMRMapper.MATRIX_FILE_NAMES_CONFIG, inputs);
		
		// configure input format
		String[] names=inputs.split(",");
		Vector<Path> paths=new Vector<Path>();
		for(String name: names)
		{
			Path p=new Path(name);
			boolean redundant=false;
			for(Path ep: paths)
				if(ep.equals(p))
				{
					redundant=true;
					break;
				}
			if(redundant)
				continue;
			FileInputFormat.addInputPath(job, p);
			paths.add(p);
		}
		
		//configure reducer
		job.setNumReduceTasks(numReducers);
		job.setInt("dfs.replication", replication);
		
		job.setLong(ABMRMapper.MATRIX_NUM_ROW_PREFIX_CONFIG+0, nr1);
		job.setLong(ABMRMapper.MATRIX_NUM_COLUMN_PREFIX_CONFIG+0, nc1);
		job.setLong(ABMRMapper.MATRIX_NUM_ROW_PREFIX_CONFIG+1, nr2);
		job.setLong(ABMRMapper.MATRIX_NUM_COLUMN_PREFIX_CONFIG+1, nc2);
		
		job.setInt(ABMRMapper.BLOCK_NUM_ROW_PREFIX_CONFIG+0, bnr1);
		job.setInt(ABMRMapper.BLOCK_NUM_COLUMN_PREFIX_CONFIG+0, bnc1);
		job.setInt(ABMRMapper.BLOCK_NUM_ROW_PREFIX_CONFIG+1, bnr2);
		job.setInt(ABMRMapper.BLOCK_NUM_COLUMN_PREFIX_CONFIG+1, bnc2);
		job.set(ABMRReducer.AGGREGATE_BINARY_OPERATION_CONFIG, aggBinaryOperation);
		// configure output
		Path outPath=new Path(output);
		FileOutputFormat.setOutputPath(job, new Path(output));
		FileOutputFormat.setOutputPath(job, outPath);
		MapReduceTool.deleteFileIfExistOnHDFS(outPath, job);
		
		job.setInputFormat(SequenceFileInputFormat.class);
		
		// configure mapper
		job.setMapperClass(ABMRMapper.class);
		job.setMapOutputKeyClass(TaggedTripleIndexes.class);
		job.setMapOutputValueClass(MatrixBlock.class);
		job.setOutputKeyComparatorClass(TaggedTripleIndexes.Comparator.class);
		job.setPartitionerClass(TaggedTripleIndexes.FirstTwoIndexesPartitioner.class);
		
		//configure reducer
		job.setReducerClass(ABMRReducer.class);
		job.setOutputKeyClass(MatrixIndexes.class);
		job.setOutputValueClass(MatrixBlock.class);
		job.setOutputFormat(SequenceFileOutputFormat.class);
	
		JobClient.runJob(job);
		
		return job;
	}
}