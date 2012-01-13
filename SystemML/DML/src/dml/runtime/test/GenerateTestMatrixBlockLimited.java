package dml.runtime.test;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;

import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.test.GenerateTestMatrixBlockLimited.SequenceOutMapper;
import dml.runtime.util.MapReduceTool;

public class GenerateTestMatrixBlockLimited
{
	public static int maxRandom = 5000;
	
	
	static class SequenceOutMapper extends MapReduceBase
			implements Mapper<LongWritable, Text, MatrixIndexes, MatrixBlock>
	{
		private MatrixIndexes indexes = new MatrixIndexes();
		private MatrixBlock block = new MatrixBlock();
		private Random random;
		
		
		@Override
		public void map(LongWritable key, Text value,
				OutputCollector<MatrixIndexes, MatrixBlock> out, Reporter report)
		throws IOException
		{
			String[] strs = value.toString().split(",");
			long i = Long.parseLong(strs[0]);
			long j = Long.parseLong(strs[1]);
			int blockRlen = Integer.parseInt(strs[2]);
			int blockClen = Integer.parseInt(strs[3]);
			double minValue = Double.parseDouble(strs[4]);
			double maxValue = Double.parseDouble(strs[5]);
			long seed = Long.parseLong(strs[6]);
			random = new Random(seed);
			double sparcity = Double.parseDouble(strs[7]);
			indexes.setIndexes(i+1, j+1);
			
			if(sparcity > 0.333)
				block.reset(blockRlen, blockClen, false);
			else
				block.reset(blockRlen, blockClen, true);
			
			double currentValue;
			for(int bi = 0; bi < blockRlen; bi++)
				for(int bj = 0; bj < blockClen; bj++)
				{
					if(random.nextDouble() > sparcity)
						continue;
					// get random number out of range
					currentValue = (double)random.nextInt(maxRandom) / (double)maxRandom;
					currentValue = (currentValue * (maxValue - minValue)) + minValue;
					block.setValue(bi, bj, currentValue);
				}
		//	if(block.getNonZeros() > 0)
		//	{
				out.collect(indexes, block);
		//	}
		}
	}
	
	public static void generateJobFile(JobConf job, Path jobFile, long numRows, 
			long numColumns, int blockRowSize, int blockColumnsSize,
			double minValue, double maxValue, long seed, double sparcity)
	throws IOException
	{
		FileSystem fs = FileSystem.get(job);
		FSDataOutputStream fsout = fs.create(jobFile);
		PrintWriter out = new PrintWriter(fsout);
		
		for(long i = 0; i < numRows; i += blockRowSize)
		{
			long realBlockRlen = Math.min(blockRowSize, numRows - i);
			for(long j = 0; j < numColumns; j += blockColumnsSize)
			{
				long realBlockClen = Math.min(blockColumnsSize, numColumns - j);
				out.println((i / blockRowSize) + "," + (j / blockColumnsSize) + "," + realBlockRlen +
						"," + realBlockClen + "," + minValue + "," + maxValue +
						"," + (seed + i * numRows + j)+ "," + sparcity);
			}
		}
		out.close();
	}
	
	public static JobConf runJob(int numMappers, long numRows, long numColumns, String outDir, 
			int replication, int blockRowSize, int blockColumnsSize, double minValue, double maxValue,
			long seed, double sparcity) 
	throws IOException
	{
		Path jobFile = new Path( "GenerateTestMatrixBlockLimited-seeds-"
				+ Integer.toString(new Random().nextInt(Integer.MAX_VALUE)));
		Path outputDir = new Path(outDir);
		// create job
		JobConf job = new JobConf(GenerateTestMatrixBlockLimited.class);
		try
		{ 
			job.setJobName("GenerateTestMatrix");
			
			// create job file
			generateJobFile(job, jobFile, numRows, numColumns, blockRowSize, blockColumnsSize,
					minValue, maxValue, seed, sparcity);
			
			// configure input
			//job.setInputFormat(MapPerLineTextInputFormat.class);
			FileInputFormat.setInputPaths(job, jobFile);
			
			// configure mappers
			job.setNumMapTasks(numMappers);
			job.setMapperClass(SequenceOutMapper.class);
			job.setMapOutputKeyClass(MatrixIndexes.class);
			job.setMapOutputValueClass(MatrixBlock.class);
			job.setOutputFormat(SequenceFileOutputFormat.class);
			
			job.setOutputKeyClass(MatrixIndexes.class);
			job.setOutputValueClass(MatrixBlock.class);
			// configure reducers
			job.setNumReduceTasks(0);
			// job.setReducerClass(IdentityReducer.class);
			
			// configure output
			FileOutputFormat.setOutputPath(job, outputDir);
			MapReduceTool.deleteFileIfExistOnHDFS(outputDir, job);
			job.setInt("dfs.replication", replication);
			
			JobClient.runJob(job);
		} finally
		{
			MapReduceTool.deleteFileIfExistOnHDFS(jobFile, job);
		}
		return job;
	}
	
	public static void main(String[] args) throws Exception
	{
		if(args.length < 11)
		{
			System.err.println("expect 11 parameters: numMappers numRows numcolumns " +
					"outputDir replication blockRowSize blockColumnSize minValue maxValue " +
					"seed sparcity");
			System.exit(1);
		}
		
		int numMappers = Integer.parseInt(args[0]);
		long numRows = Long.parseLong(args[1]);
		long numColumns = Long.parseLong(args[2]);
		String outDir = args[3];
		int replication = Integer.parseInt(args[4]);
		int blockRowSize = Integer.parseInt(args[5]);
		int blockColumnsSize = Integer.parseInt(args[6]);
		double minValue = Double.parseDouble(args[7]);
		double maxValue = Double.parseDouble(args[8]);
		long seed = Long.parseLong(args[9]);
		double sparcity = Double.parseDouble(args[10]);
		JobConf job = GenerateTestMatrixBlockLimited.runJob(numMappers, numRows, numColumns, outDir, 
				replication, blockRowSize, blockColumnsSize, minValue, maxValue, seed, sparcity);
	}

}
