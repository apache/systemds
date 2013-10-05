/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.meta;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

import org.apache.commons.math.random.Well1024a;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.ibm.bi.dml.runtime.matrix.io.MapPerLineTextInputFormat;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

public class PartitionCellMR
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	static class SequenceOutMapper extends MapReduceBase implements Mapper<LongWritable, Text, MatrixIndexes, MatrixCell> {
		MultipleOutputs multipleOutputs ;
		private Well1024a random = new Well1024a() ;
		private Text textbuf=new Text();
		int numIterations ;
		MatrixIndexes mi = new MatrixIndexes() ;
		MatrixCell mc = new MatrixCell() ;
		@Override
		public void map(LongWritable key, Text value,
				OutputCollector<MatrixIndexes, MatrixCell> out,
				Reporter reporter) throws IOException {
			
			String[] strs=value.toString().split(",");
			long from=Long.parseLong(strs[0]);
			long numRowsPerMapper =Long.parseLong(strs[1]);
			long numColumn=Long.parseLong(strs[2]);
			double sparcity=Double.parseDouble(strs[3]);
			long numNonZero = (long) (sparcity * numColumn * numRowsPerMapper) ;
			
			for(long i = 0 ; i < numNonZero; i++) {
				for(int j = 0 ; j < numIterations; j++) {
					long rowId = from + (long)(random.nextDouble() * (double) numRowsPerMapper) ;
					long colId = (long)(random.nextDouble() * (double) numColumn) ;
					mi.setIndexes(rowId, colId) ;
					mc.setValue(1) ;
					multipleOutputs.getCollector(""+j, reporter).collect(mi, mc) ;
				}
			}
		}

		public void close() throws IOException {
			multipleOutputs.close();
		}

		@Override
		public void configure(JobConf job) {
			multipleOutputs = new MultipleOutputs(job) ;
			numIterations = job.getInt("numIterations", 1) ;
			int mapperId = MapReduceTool.getUniqueMapperId(job, true) ;
			for(int i = 0 ; i < mapperId; i++){
				// DOUG: RANDOM
				//random.resetNextSubstream() ;
			}
		}
	}

	public static void generateJobFile(JobConf job, Path jobFile, int numMappers, long numRows, 
			long numColumns, double sparcity) throws IOException {
		FileSystem fs = FileSystem.get(job);
		FSDataOutputStream fsout = fs.create(jobFile);
		PrintWriter out = new PrintWriter(fsout);

		long numRowsPerMapper=(long) Math.ceil((double)numRows/(double)numMappers);
		long size = 0;
		for (int i=0; i<numMappers && size<numRows; i++) {
			long realNumRows=Math.min(numRowsPerMapper, numRows-size);

			out.println(size + "," + realNumRows + "," + numColumns+","+sparcity);
			size+=realNumRows;
		}
		out.close();
	}

	public static void runJob(int numMappers, long numRows, long numColumns, 
			int replication, PartitionParams pp) 
	throws IOException
	{
		Path jobFile = new Path( "GenerateTestMatrix-seeds-"
				+ Integer.toString(new Random().nextInt(Integer.MAX_VALUE)));
		//Path outputDir=new Path(outDir);
		// create job
		JobConf job = new JobConf(PartitionCellMR.class);
		try { 
			job.setJobName("GenerateTestMatrix");

			// create job file
			generateJobFile(job, jobFile, numMappers, numRows, numColumns, pp.frac);

			// configure input
			job.setInputFormat(MapPerLineTextInputFormat.class);
			FileInputFormat.setInputPaths(job, jobFile);

			byte[] resultIndexes = pp.getResultIndexes() ;
			byte[] resultDimsUnknown = pp.getResultDimsUnknown();
			String[] outputs = pp.getOutputStrings();
			OutputInfo[] outputInfos = new OutputInfo[outputs.length] ;
			for(int i = 0 ; i < outputInfos.length; i++){ 
				outputInfos[i] = OutputInfo.BinaryBlockOutputInfo ;
			}
			try {
				// TODO: DRB -- changed this to block
				MRJobConfiguration.setUpMultipleOutputs(job, resultIndexes, resultDimsUnknown, outputs, outputInfos, true);
			} catch (Exception e) {
				e.printStackTrace();
			}

			// configure mappers
			job.setMapperClass(SequenceOutMapper.class);
			job.setMapOutputKeyClass(MatrixIndexes.class);
			job.setMapOutputValueClass(MatrixCell.class);
			job.setInt("numIterations", pp.numIterations) ;

			// configure reducers
			job.setNumReduceTasks(0);

			// configure output
			job.setInt("dfs.replication", replication);

			JobClient.runJob(job);
		} finally {
			MapReduceTool.deleteFileIfExistOnHDFS(jobFile, job);
		}
	}
}