package dml.runtime.test;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
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
import org.apache.hadoop.mapred.TextOutputFormat;

import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.util.MapReduceTool;

public class GenerateTestMatrixLimited
{
	public static int maxRandom = 5000;
	
	
	static class SequenceOutMapper extends MapReduceBase
			implements Mapper<LongWritable, Text, MatrixIndexes, DoubleWritable>
	{
		private MatrixIndexes indexes = new MatrixIndexes();
		private DoubleWritable cellValue = new DoubleWritable();
		private Random random;
		
		
		@Override
		public void map(LongWritable key, Text value,
				OutputCollector<MatrixIndexes, DoubleWritable> out, Reporter report)
		throws IOException
		{
			String[] strs = value.toString().split(",");
			long from = Long.parseLong(strs[0]);
			long length = Long.parseLong(strs[1]);
			long numColumn = Long.parseLong(strs[2]);
			double minValue = Double.parseDouble(strs[3]);
			double maxValue = Double.parseDouble(strs[4]);
			long seed = Long.parseLong(strs[5]);
			random = new Random(seed);
			double sparcity = Double.parseDouble(strs[6]);
			
			double currentValue;
			for(long r = 0; r < length; r++)
			{
				for(long c = 0; c < numColumn; c++)
				{
					if(random.nextDouble() > sparcity)
						continue;
					indexes.setIndexes(r + from, c);
					currentValue = (double) random.nextInt(maxRandom) / (double)maxRandom;
					currentValue = (currentValue % (maxValue - minValue)) + minValue;
					cellValue.set(currentValue);
					out.collect(indexes, cellValue);
				}
			}
		}
	}
	
	static class TextOutMapper extends MapReduceBase
			implements Mapper<LongWritable, Text, NullWritable, Text>
	{
		private Random random;
		private Text textbuf = new Text();
		
		
		@Override
		public void map(LongWritable key, Text value,
				OutputCollector<NullWritable, Text> out, Reporter report)
		throws IOException
		{
			String[] strs = value.toString().split(",");
			long from = Long.parseLong(strs[0]);
			long length = Long.parseLong(strs[1]);
			long numColumn = Long.parseLong(strs[2]);
			double minValue = Double.parseDouble(strs[3]);
			double maxValue = Double.parseDouble(strs[4]);
			long seed = Long.parseLong(strs[5]);
			random = new Random(seed);
			double sparcity = Double.parseDouble(strs[6]);
			
			double currentValue;
			for(long r = 0; r < length; r++)
			{
				for(long c = 0; c < numColumn; c++)
				{
					if(random.nextDouble() > sparcity)
						continue;
					// get random number out of range
					currentValue = (double) random.nextInt(maxRandom) / (double)maxRandom;
					currentValue = (currentValue % (maxValue - minValue)) + minValue;
					textbuf.set((r + from) + " " + c + " " + currentValue);
					out.collect(NullWritable.get(), textbuf);
				}
			}
		}
	}
	
	public static void generateJobFile(JobConf job, Path jobFile, int numMappers, long numRows, 
			long numColumns, double minValue, double maxValue, long seed, double sparcity)
	throws IOException
	{
		FileSystem fs = FileSystem.get(job);
		FSDataOutputStream fsout = fs.create(jobFile);
		PrintWriter out = new PrintWriter(fsout);
		
		long numRowsPerMapper = (long) Math.ceil((double) numRows / (double) numMappers);
		long size = 0;
		for (int i = 0; i < numMappers && size < numRows; i++)
		{
			long realNumRows = Math.min(numRowsPerMapper, numRows - size);
			out.println(size + "," + realNumRows + "," + numColumns + "," + minValue + "," + maxValue +
					"," + (seed + i) + "," + sparcity);
			size += realNumRows;
		}
		out.close();
	}
	
	public static void runJob(int numMappers, long numRows, long numColumns, String outDir, 
			int replication, double minValue, double maxValue, boolean isSequenceOut,
			long seed, double sparcity) 
	throws IOException
	{
		Path jobFile = new Path("GenerateTestMatrixLimited-seeds-" +
				Integer.toString(new Random().nextInt(Integer.MAX_VALUE)));
		Path outputDir = new Path(outDir);
		// create job
		JobConf job = new JobConf(GenerateTestMatrixLimited.class);
		try
		{ 
			job.setJobName("GenerateTestMatrix");
			
			// create job file
			generateJobFile(job, jobFile, numMappers, numRows, numColumns, minValue, maxValue, seed, sparcity);
			
			// configure input
			job.setInputFormat(MapPerLineTextInputFormat.class);
			FileInputFormat.setInputPaths(job, jobFile);
			
			// configure mappers
			if(isSequenceOut)
			{
				job.setMapperClass(SequenceOutMapper.class);
				job.setMapOutputKeyClass(MatrixIndexes.class);
				job.setMapOutputValueClass(DoubleWritable.class);
				job.setOutputFormat(SequenceFileOutputFormat.class);
				job.setOutputKeyClass(MatrixIndexes.class);
				job.setOutputValueClass(DoubleWritable.class);
			} else
			{
				job.setMapperClass(TextOutMapper.class);
				job.setMapOutputKeyClass(NullWritable.class);
				job.setMapOutputValueClass(Text.class);
				job.setOutputFormat(TextOutputFormat.class);
				job.setOutputKeyClass(NullWritable.class);
				job.setOutputValueClass(Text.class);
			}
			
			// configure reducers
			job.setNumReduceTasks(0);
			
			// configure output
			FileOutputFormat.setOutputPath(job, outputDir);
			MapReduceTool.deleteFileIfExistOnHDFS(outputDir, job);
			job.setInt("dfs.replication", replication);
			
			JobClient.runJob(job);
		} finally
		{
			MapReduceTool.deleteFileIfExistOnHDFS(jobFile, job);
		}
	}
	
	public static void main(String[] args) throws Exception
	{
		if(args.length < 10)
		{
			System.err.println("expect 10 paramers: numMappers numRows numColumns " +
					"outputDir replication minValue maxValue sequenceOutputFile? seed sparcity");
			System.exit(-1);
		}
		
		int numMappers = Integer.parseInt(args[0]);
		long numRows = Long.parseLong(args[1]);
		long numColumns = Long.parseLong(args[2]);
		String outputDir = args[3];
		int replication = Integer.parseInt(args[4]);
		double minValue = Double.parseDouble(args[5]);
		double maxValue = Double.parseDouble(args[6]);
		boolean isSequenceOut = Boolean.parseBoolean(args[7]);
		long seed = Long.parseLong(args[8]);
		double sparcity = Double.parseDouble(args[9]);
		GenerateTestMatrixLimited.runJob(numMappers, numRows, numColumns, outputDir, 
				replication, minValue, maxValue, isSequenceOut, seed, sparcity);
	}
}
