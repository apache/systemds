package gnmf.comperator;

import gnmf.io.MatrixObject;
import gnmf.io.MatrixVector;
import gnmf.io.TaggedIndex;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
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
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;


public class WHComperator
{
	protected static enum Counters { WRONG_CELLS };
	
	static class WHComperatorMapper extends MapReduceBase
			implements Mapper<TaggedIndex, MatrixObject, TaggedIndex, MatrixVector>
	{
		@Override
		public void map(TaggedIndex key, MatrixObject value,
				OutputCollector<TaggedIndex, MatrixVector> out, Reporter reporter)
				throws IOException
		{
			if(key.getType() == TaggedIndex.TYPE_VECTOR)
				out.collect(key, (MatrixVector) value.getObject());
		}
	}
	
	static class WHComperatorReducer extends MapReduceBase
			implements Reducer<TaggedIndex, MatrixVector, NullWritable, Text>
	{
		private float epsilon;
		private JobConf job;
		
		
		@Override
		public void reduce(TaggedIndex key, Iterator<MatrixVector> values,
				OutputCollector<NullWritable, Text> out, Reporter reporter)
				throws IOException
		{
			double max = 0;
			double min = epsilon;
			double sum = 0;
			
			if(!values.hasNext())
				throw new RuntimeException("expected two vectors");
			double[] vector1 = values.next().getValues();
			if(!values.hasNext())
				throw new RuntimeException("expected two vectors");
			double[] vector2 = values.next().getValues();
			if(vector1.length != vector2.length)
				throw new RuntimeException("vectors of different length");
			for(int i = 0; i < vector1.length; i++)
			{
				double diff = Math.abs(vector1[i] - vector2[i]);
				if(diff > max)
					max = diff;
				if(diff < min)
					min = diff;
				sum += diff;
				if(diff > epsilon)
				{
					out.collect(null, new Text(vector1[i] + " " + vector2[i]));
					reporter.incrCounter(Counters.WRONG_CELLS, 1);
				}
			}
			
			if(max > job.getFloat("gnmf.comperator.max", 0))
				job.setFloat("gnmf.comperator.max", (float) max);
			if(min < job.getFloat("gnmf.comperator.min", epsilon))
				job.setFloat("gnmf.comperator.min", (float) min);
			job.setFloat("gnmf.comperator.sum", (float) (sum +
					job.getFloat("gnmf.comperator.sum", 0)));
			job.setLong("gnmf.comperator.cells", job.getLong("gnmf.comperator.cells", 0) +
					vector1.length);
		}
		
		@Override
		public void configure(JobConf job)
		{
			epsilon = job.getFloat("gnmf.comperator.epsilon", 0.0F);
			this.job = job;
		}
	}
	
	public static void runJob(int numMappers, int numReducers, int replication,
			String dmlInputDir, String rawInputDir, float epsilon, String workingDir)
	throws IOException
	{
		String workingDirectory = workingDir + System.currentTimeMillis() + "-WHComperator/";

		JobConf job = new JobConf(WHComperator.class);
		job.setJobName("WHComperator");
		
		job.setInputFormat(SequenceFileInputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(dmlInputDir), new Path(rawInputDir));
		
		job.setOutputFormat(TextOutputFormat.class);
		FileOutputFormat.setOutputPath(job, new Path(workingDirectory));
		
		job.setNumMapTasks(numMappers);
		job.setMapperClass(WHComperatorMapper.class);
		job.setMapOutputKeyClass(TaggedIndex.class);
		job.setMapOutputValueClass(MatrixVector.class);
		
		job.setNumReduceTasks(numReducers);
		job.setReducerClass(WHComperatorReducer.class);
		job.setOutputKeyClass(NullWritable.class);
		job.setOutputValueClass(Text.class);
		
		job.setFloat("gnmf.comperator.epsilon", epsilon);
		job.setFloat("gnmf.comperator.min", epsilon);
		job.setFloat("gnmf.comperator.max", 0);
		job.setFloat("gnmf.comperator.sum", 0);
		job.setLong("gnmf.comperator.cells", 0);
		
		JobClient.runJob(job);
		
		System.out.println("min difference: " +
				job.getFloat("gnmf.comperator.min", epsilon));
		System.out.println("max difference: " +
				job.getFloat("gnmf.comperator.max", -1));
		double variance = (job.getFloat("gnmf.comperator.sum", 0) /
				job.getLong("gnmf.comperator.cells", 1));
		System.out.println("variance: " + variance);
	}
	
	public static void main(String[] args) throws IOException
	{
		if(args.length < 7)
		{
			System.out.println("expected parameters: [dml input dir] [raw input dir] [epsilon] " +
					"[working dir] [num mappers] [num reducers] [replication]");
			System.exit(1);
		}
		
		String dmlInputDir = args[0];
		String rawInputDir = args[1];
		float epsilon = Float.parseFloat(args[2]);
		String workingDir = args[3];
		int numMappers = Integer.parseInt(args[4]);
		int numReducers = Integer.parseInt(args[5]);
		int replication = Integer.parseInt(args[6]);
		WHComperator.runJob(numMappers, numReducers, replication, dmlInputDir, rawInputDir,
				epsilon, workingDir);
	}
}
