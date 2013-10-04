package gnmf;

import gnmf.io.MatrixCell;
import gnmf.io.MatrixFormats;
import gnmf.io.MatrixObject;
import gnmf.io.MatrixVector;
import gnmf.io.TaggedIndex;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
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


public class UpdateWHStep1
{
	public static final int UPDATE_TYPE_H = 0;
	public static final int UPDATE_TYPE_W = 1;
	
	static class UpdateWHStep1Mapper extends MapReduceBase
			implements Mapper<TaggedIndex, MatrixObject, TaggedIndex, MatrixObject>
	{
		private int updateType;
		
		@Override
		public void map(TaggedIndex key, MatrixObject value,
				OutputCollector<TaggedIndex, MatrixObject> out,
				Reporter reporter) throws IOException
		{
			if(updateType == UPDATE_TYPE_W && key.getType() == TaggedIndex.TYPE_CELL)
			{
				MatrixCell current = (MatrixCell) value.getObject();
				out.collect(new TaggedIndex(current.getColumn(), TaggedIndex.TYPE_CELL),
						new MatrixObject(new MatrixCell(key.getIndex(), current.getValue())));
			} else
			{
				out.collect(key, value);
			}
		}
		
		@Override
		public void configure(JobConf job)
		{
			updateType = job.getInt("gnmf.updateType", 0);
		}
	}
	
	static class UpdateWHStep1Reducer extends MapReduceBase
			implements Reducer<TaggedIndex, MatrixObject, TaggedIndex, MatrixVector>
	{
		private double[] baseVector = null;
		private int vectorSizeK;
		
		
		@Override
		public void reduce(TaggedIndex key, Iterator<MatrixObject> values,
				OutputCollector<TaggedIndex, MatrixVector> out, Reporter reporter)
				throws IOException
		{
			if(key.getType() == TaggedIndex.TYPE_VECTOR)
			{
				if(!values.hasNext())
					throw new RuntimeException("expected vector");
				MatrixFormats current = values.next().getObject();
				if(!(current instanceof MatrixVector))
					throw new RuntimeException("expected vector");
				baseVector = ((MatrixVector) current).getValues();
			} else
			{
				while(values.hasNext())
				{
					MatrixCell current = (MatrixCell) values.next().getObject();
					if(baseVector == null)
					{
						out.collect(new TaggedIndex(current.getColumn(), TaggedIndex.TYPE_VECTOR),
								new MatrixVector(vectorSizeK));
					} else
					{
						if(baseVector.length == 0)
							throw new RuntimeException("base vector is corrupted");
						MatrixVector resultingVector = new MatrixVector(baseVector);
						resultingVector.multiplyWithScalar(current.getValue());
						if(resultingVector.getValues().length == 0)
							throw new RuntimeException("multiplying with scalar failed");
						out.collect(new TaggedIndex(current.getColumn(), TaggedIndex.TYPE_VECTOR),
								resultingVector);
					}
				}
				baseVector = null;
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
	
	public static String runJob(int numMappers, int numReducers, int replication,
			int updateType, String matrixInputDir, String whInputDir, String outputDir,
			int k) throws IOException
	{
		String workingDirectory = outputDir + System.currentTimeMillis() + "-UpdateWHStep1/";

		JobConf job = new JobConf(UpdateWHStep1.class);
		job.setJobName("MatrixGNMFUpdateWHStep1");
		
		job.setInputFormat(SequenceFileInputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(matrixInputDir), new Path(whInputDir));
		
		job.setOutputFormat(SequenceFileOutputFormat.class);
		FileOutputFormat.setOutputPath(job, new Path(workingDirectory));
		
		job.setNumMapTasks(numMappers);
		job.setMapperClass(UpdateWHStep1Mapper.class);
		job.setMapOutputKeyClass(TaggedIndex.class);
		job.setMapOutputValueClass(MatrixObject.class);
		
		job.setNumReduceTasks(numReducers);
		job.setReducerClass(UpdateWHStep1Reducer.class);
		job.setOutputKeyClass(TaggedIndex.class);
		job.setOutputValueClass(MatrixVector.class);
		
		job.setInt("gnmf.updateType", updateType);
		job.setInt("dml.matrix.gnmf.k", k);
		
		JobClient.runJob(job);
		
		return workingDirectory;
	}
}
