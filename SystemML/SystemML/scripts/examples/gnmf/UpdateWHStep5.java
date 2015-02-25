package gnmf;

import gnmf.io.MatrixFormats;
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


public class UpdateWHStep5
{
	static class UpdateWHStep5Mapper extends MapReduceBase
			implements Mapper<TaggedIndex, MatrixObject, TaggedIndex, MatrixVector>
	{
		@Override
		public void map(TaggedIndex key, MatrixObject value,
				OutputCollector<TaggedIndex, MatrixVector> out, Reporter reporter)
				throws IOException
		{
			out.collect(key, (MatrixVector) value.getObject());
		}
	}
	
	static class UpdateWHStep5Reducer extends MapReduceBase
			implements Reducer<TaggedIndex, MatrixVector, TaggedIndex, MatrixObject>
	{
		private MatrixVector result	= null;
		private boolean multiplied	= false;
		private int vectorSizeK;
		
		
		@Override
		public void reduce(TaggedIndex key, Iterator<MatrixVector> values,
				OutputCollector<TaggedIndex, MatrixObject> out, Reporter reporter)
				throws IOException
		{
			MatrixVector current = values.next();
			if(result == null)
			{
				if(key.getType() == TaggedIndex.TYPE_VECTOR_HW)
				{
					result = current.getCopy();
				} else
				{
					result = new MatrixVector(vectorSizeK);
					if(key.getType() == TaggedIndex.TYPE_VECTOR_Y)
					{
						out.collect(new TaggedIndex(key.getIndex(), TaggedIndex.TYPE_VECTOR),
								new MatrixObject(result));
						result = null;
					}
				}
			} else if(!multiplied)
			{
				if(key.getType() == TaggedIndex.TYPE_VECTOR_X)
				{
					result.elementWiseMultiplication(current);
					multiplied = true;
				} else
				{
					result = new MatrixVector(vectorSizeK);
					if(key.getType() == TaggedIndex.TYPE_VECTOR_Y)
					{
						out.collect(new TaggedIndex(key.getIndex(), TaggedIndex.TYPE_VECTOR),
								new MatrixObject(result));
						result = null;
						multiplied = false;
					} else
					{
						throw new RuntimeException();
					}
				}
			} else
			{
				if(key.getType() == TaggedIndex.TYPE_VECTOR_Y)
				{
					result.elementWiseDivision(current);
					out.collect(new TaggedIndex(key.getIndex(), TaggedIndex.TYPE_VECTOR),
							new MatrixObject(result));
					result = null;
					multiplied = false;
				} else
				{
					throw new RuntimeException();
				}
			}
		}
		
		@Override
		public void configure(JobConf job)
		{
			vectorSizeK = job.getInt("dml.matrix.gnmf.k", 0);
			if(vectorSizeK == 0)
				throw new RuntimeException("invalid k specified");
		}
	}
	
	public static String runJob(int numMappers, int numReducers, int replication, String hwInputDir,
			String xInputDir, String yInputDir, String outputDir, int k) throws IOException
	{
		String workingDirectory = outputDir;
		//String workingDirectory = outputDir + System.currentTimeMillis() + "-UpdateWHStep5/";

		JobConf job = new JobConf(UpdateWHStep5.class);
		job.setJobName("MatrixGNMFUpdateWHStep5");
		
		job.setInputFormat(SequenceFileInputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(hwInputDir), new Path(xInputDir),
				new Path(yInputDir));
		
		job.setOutputFormat(SequenceFileOutputFormat.class);
		FileOutputFormat.setOutputPath(job, new Path(workingDirectory));
		
		job.setNumMapTasks(numMappers);
		job.setMapperClass(UpdateWHStep5Mapper.class);
		job.setMapOutputKeyClass(TaggedIndex.class);
		job.setMapOutputValueClass(MatrixVector.class);
		
		job.setNumReduceTasks(numReducers);
		job.setReducerClass(UpdateWHStep5Reducer.class);
		job.setOutputKeyClass(TaggedIndex.class);
		job.setOutputValueClass(MatrixObject.class);

		job.setInt("dml.matrix.gnmf.k", k);
		
		JobClient.runJob(job);
		
		return workingDirectory;
	}
}
