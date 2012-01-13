package gnmf;

import gnmf.UpdateWHStep2.UpdateWHStep2Mapper;
import gnmf.UpdateWHStep2.UpdateWHStep2Reducer;
import gnmf.io.MatrixFormats;
import gnmf.io.MatrixObject;
import gnmf.io.MatrixVector;
import gnmf.io.SquareMatrix;
import gnmf.io.TaggedIndex;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Iterator;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
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
import org.apache.hadoop.security.UnixUserGroupInformation;
import org.apache.hadoop.util.StringUtils;


public class UpdateWHStep3
{
	static class UpdateWHStep3Mapper extends MapReduceBase
			implements Mapper<TaggedIndex, MatrixObject, IntWritable, SquareMatrix>
	{
		@Override
		public void map(TaggedIndex key, MatrixObject value,
				OutputCollector<IntWritable, SquareMatrix> out, Reporter reporter)
				throws IOException
		{
			out.collect(new IntWritable(0),
					new SquareMatrix(((MatrixVector) value.getObject())));
		}
	}
	
	static class UpdateWHStep3Reducer extends MapReduceBase
			implements Reducer<IntWritable, SquareMatrix, TaggedIndex, MatrixObject>
	{
		@Override
		public void reduce(IntWritable key, Iterator<SquareMatrix> values,
				OutputCollector<TaggedIndex, MatrixObject> out, Reporter reporter)
				throws IOException
		{
			SquareMatrix result = null;
			while(values.hasNext())
			{
				SquareMatrix current = values.next();
				if(result == null)
					result = current.getCopy();
				else
					result.addMatrix(current);
			}
			out.collect(new TaggedIndex(0, TaggedIndex.TYPE_MATRIX),
					new MatrixObject(result));
		}
	}
	
	public static String runJob(int numMappers, int numReducers, int replication,
			String inputDir, String outputDir) throws IOException, URISyntaxException
	{
		String workingDirectory = outputDir + System.currentTimeMillis() + "-UpdateWHStep3/";

		JobConf job = new JobConf(UpdateWHStep3.class);
		job.setJobName("MatrixGNMFUpdateWHStep3");
		
		job.setInputFormat(SequenceFileInputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(inputDir));
		
		job.setOutputFormat(SequenceFileOutputFormat.class);
		FileOutputFormat.setOutputPath(job, new Path(workingDirectory));
		
		job.setNumMapTasks(numMappers);
		job.setMapperClass(UpdateWHStep3Mapper.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(SquareMatrix.class);
		
		job.setNumReduceTasks(numReducers);
		job.setReducerClass(UpdateWHStep3Reducer.class);
		job.setOutputKeyClass(TaggedIndex.class);
		job.setOutputValueClass(MatrixObject.class);
		
		JobClient.runJob(job);
	    
		return workingDirectory;
	}
}
