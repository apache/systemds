package gnmf.generation;

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
public class GenerateTestMatrix{

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
			double sparcity=Double.parseDouble(strs[4]);
			for(long r=0; r<length; r++)
			{
				for(long c=0; c<numColumn; c++)
				{
					if(random.nextDouble()>sparcity)
						continue;
					textbuf.set((r+from)+" "+c+" "+((double)random.nextInt(maxRandom)/(double)maxRandom));
					out.collect(NullWritable.get(), textbuf);
				}
			}
		}
	}
	
	public static void generateJobFile(JobConf job, Path jobFile, int numMappers, long numRows, 
			long numColumns, long seed, double sparcity) throws IOException {
		FileSystem fs = FileSystem.get(job);
		FSDataOutputStream fsout = fs.create(jobFile);
		PrintWriter out = new PrintWriter(fsout);
		
		long numRowsPerMapper=(long) Math.ceil((double)numRows/(double)numMappers);
		long size = 0;
		for (int i=0; i<numMappers && size<numRows; i++) {
			long realNumRows=Math.min(numRowsPerMapper, numRows-size);
			
			out.println(size + "," + realNumRows + "," + numColumns+","+(seed+i)+","+sparcity);
			size+=realNumRows;
		}
		out.close();
	}
	
	public static void runJob(int numMappers, long numRows, long numColumns, String outDir, 
			int replication, boolean isSequenceOut, long seed, double sparcity) 
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
	
	public static void main(String[] args) throws Exception {
		if(args.length<8)
		{
			System.err.println("expect 8 paramers: numMappers numRows numColumns " +
					"outputDir replication sequenceOutputFile? seed sparcity");
			return;
		}
		
		int numMappers=Integer.parseInt(args[0]);
		long numRows=Long.parseLong(args[1]);
		long numColumns=Long.parseLong(args[2]);
		String outDir=args[3];
		int replication=Integer.parseInt(args[4]);
		boolean isSequenceOut=Boolean.parseBoolean(args[5]);
		long seed=Long.parseLong(args[6]);
		double sparcity=Double.parseDouble(args[7]);
		GenerateTestMatrix.runJob(numMappers, numRows, numColumns, outDir, 
				replication, isSequenceOut, seed, sparcity);
	}
}
