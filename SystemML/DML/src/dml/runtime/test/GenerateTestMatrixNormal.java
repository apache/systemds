package dml.runtime.test;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Iterator;
import java.util.Random;

import org.apache.commons.math.random.RandomData;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
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
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;

import dml.runtime.matrix.io.MatrixCell;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.util.MapReduceTool;

public class GenerateTestMatrixNormal {

	public static int maxRandom=5000;
	static class SequenceOutMapper extends MapReduceBase implements Mapper<LongWritable, Text, 
	MatrixIndexes, MatrixCell>
	{
		private MatrixIndexes indexes=new MatrixIndexes();
		private MatrixCell cellValue=new MatrixCell();
		private Random random;
		@Override
		public void map(LongWritable key, Text value,
				OutputCollector<MatrixIndexes, MatrixCell> out,
				Reporter report) throws IOException {
			String[] strs=value.toString().split(",");
			long realNum=Long.parseLong(strs[0]);
			long numRows=Long.parseLong(strs[1]);
			long numColumn=Long.parseLong(strs[2]);
			long seed=Long.parseLong(strs[3]);
		
			//RandomData eng=RandomEngine.makeDefault();
			//Normal dist_r=new Normal((double)numRows/2, (double)numRows/6, eng);
			//Normal dist_c=new Normal((double)numColumn/2, (double)numColumn/6, eng);
			random=new Random(seed);
			for(int i=0; i<realNum; i++)
			{
				long r, c;
				do{
					// DOUG: RANDOM ISSUE
					r=0 ; //(long)dist_r.nextDouble();
				}while(r<0 || r>= numRows);
				do{
					// DOUG: RANDOM ISSUE
					c=0; //(long)dist_c.nextDouble();
				}while(c<0 || c>=numColumn);
				
				indexes.setIndexes(r, c);
			//	System.out.println(r+", "+c);
				cellValue.setValue((double)random.nextInt(maxRandom)/(double)maxRandom);
				out.collect(indexes, cellValue);
			}

		}
	}
	
	static class InnerReducer extends MapReduceBase implements Reducer<MatrixIndexes, MatrixCell, MatrixIndexes, MatrixCell>
	{

		@Override
		public void reduce(MatrixIndexes indexes, Iterator<MatrixCell> values,
				OutputCollector<MatrixIndexes, MatrixCell> out, Reporter report)
				throws IOException {
			out.collect(indexes, values.next());
		}
		
	}
	
	public static void generateJobFile(JobConf job, Path jobFile, int numMappers, long numRows, 
			long numColumns, long seed, double sparcity) throws IOException {
		FileSystem fs = FileSystem.get(job);
		FSDataOutputStream fsout = fs.create(jobFile);
		PrintWriter out = new PrintWriter(fsout);
		long totalSize=(long)(numRows*numColumns*sparcity);
		long numPerMapper=(long) Math.ceil((double)(totalSize)/(double)numMappers);
		long size = 0;
		for (int i=0; i<numMappers && size<totalSize; i++) {
			long realNum=Math.min(numPerMapper, totalSize-size);
			
			out.println(realNum + "," + numRows+ "," + numColumns+","+(seed+i));
			size+=realNum;
		}
		out.close();
	}
	
	public static void runJob(int numMappers, long numRows, long numColumns, String outDir, 
			int replication, boolean isSequenceOut, long seed, double sparcity, int numReducers) 
	throws IOException
	{
		Path jobFile = new Path( "GenerateTestMatrix-seeds-"
				+ Integer.toString(new Random().nextInt(Integer.MAX_VALUE)));
		Path outputDir=new Path(outDir);
		// create job
		JobConf job = new JobConf(GenerateTestMatrix.class);
		try { 
			job.setJobName("GenerateTestMatrix");
			
			// create job file
			generateJobFile(job, jobFile, numMappers, numRows, numColumns, seed, sparcity);
			
			// configure input
			job.setInputFormat(MapPerLineTextInputFormat.class);
			FileInputFormat.setInputPaths(job, jobFile);
			
			// configure mappers
			
			job.setMapperClass(SequenceOutMapper.class);
			job.setMapOutputKeyClass(MatrixIndexes.class);
			job.setMapOutputValueClass(MatrixCell.class);
			job.setReducerClass(InnerReducer.class);
			job.setOutputFormat(SequenceFileOutputFormat.class);
			job.setOutputKeyClass(MatrixIndexes.class);
			job.setOutputValueClass(MatrixCell.class);
			//job.setOutputFormat(TextOutputFormat.class);
			
			// configure reducers
			job.setNumReduceTasks(numReducers);
			
			// configure output
			FileOutputFormat.setOutputPath(job, outputDir);
			MapReduceTool.deleteFileIfExistOnHDFS(outputDir, job);
			job.setInt("dfs.replication", replication);
			
			JobClient.runJob(job);
		} finally {
			MapReduceTool.deleteFileIfExistOnHDFS(jobFile, job);
		}
	}
	
	public static void main(String[] args) throws Exception {
		if(args.length<9)
		{
			System.err.println("expect 8 paramers: numMappers numRows numColumns " +
					"outputDir replication sequenceOutputFile? seed sparcity numreducer");
			System.exit(-1);
		}
		
		int numMappers=Integer.parseInt(args[0]);
		long numRows=Long.parseLong(args[1]);
		long numColumns=Long.parseLong(args[2]);
		String outDir=args[3];
		int replication=Integer.parseInt(args[4]);
		boolean isSequenceOut=Boolean.parseBoolean(args[5]);
		long seed=Long.parseLong(args[6]);
		double sparcity=Double.parseDouble(args[7]);
		int numReducers=Integer.parseInt(args[8]);
		GenerateTestMatrixNormal.runJob(numMappers, numRows, numColumns, outDir, 
				replication, isSequenceOut, seed, sparcity, numReducers);
	}
}
