package gnmf;

import gnmf.io.MatrixObject;
import gnmf.io.MatrixVector;
import gnmf.io.TaggedIndex;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.fs.Path;
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


public class UpdateWHStep2
{
	static class UpdateWHStep2Mapper extends MapReduceBase
			implements Mapper<TaggedIndex, MatrixVector, TaggedIndex, MatrixVector>
	{
		@Override
		public void map(TaggedIndex key, MatrixVector value,
				OutputCollector<TaggedIndex, MatrixVector> out,
				Reporter reporter) throws IOException
		{
			out.collect(key, value);
		}
	}
	
	static class UpdateWHStep2Reducer extends MapReduceBase
			implements Reducer<TaggedIndex, MatrixVector, TaggedIndex, MatrixObject>
	{
		@Override
		public void reduce(TaggedIndex key, Iterator<MatrixVector> values,
				OutputCollector<TaggedIndex, MatrixObject> out, Reporter reporter)
				throws IOException
		{
			MatrixVector result = null;
			while(values.hasNext())
			{
				MatrixVector current = values.next();
				if(result == null)
				{
					result = current.getCopy();
				} else
				{
					result.addVector(current);
				}
			}
			if(result != null)
			{
				out.collect(new TaggedIndex(key.getIndex(), TaggedIndex.TYPE_VECTOR_X),
						new MatrixObject(result));
			}
		}
	}
	
	public static String runJob(int numMappers, int numReducers, int replication,
			String inputDir, String outputDir) throws IOException
	{
		String workingDirectory = outputDir + System.currentTimeMillis() + "-UpdateWHStep2/";

		JobConf job = new JobConf(UpdateWHStep2.class);
		job.setJobName("MatrixGNMFUpdateWHStep2");
		
		job.setInputFormat(SequenceFileInputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(inputDir));
		
		job.setOutputFormat(SequenceFileOutputFormat.class);
		FileOutputFormat.setOutputPath(job, new Path(workingDirectory));
		
		job.setNumMapTasks(numMappers);
		job.setMapperClass(UpdateWHStep2Mapper.class);
		job.setMapOutputKeyClass(TaggedIndex.class);
		job.setMapOutputValueClass(MatrixVector.class);
		
		job.setNumReduceTasks(numReducers);
		job.setReducerClass(UpdateWHStep2Reducer.class);
		job.setOutputKeyClass(TaggedIndex.class);
		job.setOutputValueClass(MatrixObject.class);
		
		JobClient.runJob(job);
		
		return workingDirectory;
	}
}
