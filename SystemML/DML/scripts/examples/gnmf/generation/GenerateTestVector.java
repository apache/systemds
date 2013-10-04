package gnmf.generation;

import gnmf.generation.GenerateTestMatrix.TextOutMapper;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

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
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextOutputFormat;


public class GenerateTestVector
{
	public static int maxRandom=5000;
	
	static class TextOutMapper extends MapReduceBase implements Mapper<LongWritable, Text, NullWritable, Text>
	{
		private Random random;
		private Text textbuf=new Text();
		@Override
		public void map(LongWritable key, Text value,
				OutputCollector<NullWritable, Text> out,
				Reporter report) throws IOException {
			String[] strs=value.toString().split(",");
			long from=Long.parseLong(strs[0]);
			long length=Long.parseLong(strs[1]);
			long numColumn=Long.parseLong(strs[2]);
			long seed=Long.parseLong(strs[3]);
			random=new Random(seed);
			for(long r=0; r<length; r++)
			{
				textbuf.set("" + (r + from));
				for(long c=0; c<numColumn; c++)
				{
					textbuf.set(textbuf.toString() + " " + ((double)random.nextInt(maxRandom)/(double)maxRandom));
				}
				out.collect(NullWritable.get(), textbuf);
			}
		}
	}
	
	public static void generateJobFile(JobConf job, Path jobFile, int numMappers, long numRows, 
			long numColumns, long seed) throws IOException {
		FileSystem fs = FileSystem.get(job);
		FSDataOutputStream fsout = fs.create(jobFile);
		PrintWriter out = new PrintWriter(fsout);
		
		long numRowsPerMapper=(long) Math.ceil((double)numRows/(double)numMappers);
		long size = 0;
		for (int i=0; i<numMappers && size<numRows; i++) {
			long realNumRows=Math.min(numRowsPerMapper, numRows-size);
			
			out.println(size + "," + realNumRows + "," + numColumns+","+(seed+i));
			size+=realNumRows;
		}
		out.close();
	}
	
	public static void runJob(int numMappers, long numRows, long numColumns, String outDir, 
			int replication, boolean isSequenceOut, long seed) 
	throws IOException
	{
		Path jobFile = new Path( "GenerateTestVector-seeds-"
				+ Integer.toString(new Random().nextInt(Integer.MAX_VALUE)));
		Path outputDir=new Path(outDir);
		// create job
		JobConf job = new JobConf(GenerateTestMatrix.class);
		try { 
			job.setJobName("GenerateTestMatrix");
			
			// create job file
			generateJobFile(job, jobFile, numMappers, numRows, numColumns, seed);
			
			// configure input
			job.setInputFormat(MapPerLineTextInputFormat.class);
			FileInputFormat.setInputPaths(job, jobFile);
			
			// configure mappers
			
			job.setMapperClass(TextOutMapper.class);
			job.setMapOutputKeyClass(NullWritable.class);
			job.setMapOutputValueClass(Text.class);
			job.setOutputFormat(TextOutputFormat.class);
			job.setOutputKeyClass(NullWritable.class);
			job.setOutputValueClass(Text.class);
			
			// configure reducers
			job.setNumReduceTasks(0);
			
			// configure output
			FileOutputFormat.setOutputPath(job, outputDir);
			if(FileSystem.get(job).exists(outputDir))
				FileSystem.get(job).delete(outputDir);
			job.setInt("dfs.replication", replication);
			
			JobClient.runJob(job);
		} finally {
			FileSystem.get(job).delete(jobFile);
		}
	}
}
