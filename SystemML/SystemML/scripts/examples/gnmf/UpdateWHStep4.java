package gnmf;

import gnmf.io.MatrixFormats;
import gnmf.io.MatrixObject;
import gnmf.io.MatrixVector;
import gnmf.io.SquareMatrix;
import gnmf.io.TaggedIndex;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.net.URI;
import java.util.Iterator;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
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


public class UpdateWHStep4
{
	public static final int UPDATE_TYPE_H = 0;
	public static final int UPDATE_TYPE_W = 1;
	
	static class UpdateWHStep4Mapper extends MapReduceBase
			implements Mapper<TaggedIndex, MatrixObject, TaggedIndex, MatrixObject>
	{
		private int updateType;
		private SquareMatrix matrixC = null;
		
		
		@Override
		public void map(TaggedIndex key, MatrixObject value,
				OutputCollector<TaggedIndex, MatrixObject> out, Reporter reporter)
				throws IOException
		{
			if(matrixC == null)
				throw new RuntimeException("where the hell is the matrix C?");
			MatrixVector current = (MatrixVector) value.getObject();
			if(updateType == UPDATE_TYPE_H)
			{
				MatrixVector v = matrixC.getCopy().multiplyWithVectorSecond(current);
				out.collect(new TaggedIndex(key.getIndex(), TaggedIndex.TYPE_VECTOR_Y),
						new MatrixObject(v));
			} else if(updateType == UPDATE_TYPE_W)
			{
				MatrixVector v = matrixC.getCopy().multiplyWithVectorFirst(current);
				out.collect(new TaggedIndex(key.getIndex(), TaggedIndex.TYPE_VECTOR_Y),
						new MatrixObject(v));
			}
		}
		
		@Override
		public void configure(JobConf job)
		{
			super.configure(job);
			updateType = job.getInt("gnmf.updateType", 0);
			try
			{
				URI[] cachedFiles = DistributedCache.getCacheFiles(job);
				if(cachedFiles == null)
					throw new IOException("matrix missing");
				SequenceFile.Reader reader = new SequenceFile.Reader(FileSystem.get(job),
						new Path(cachedFiles[cachedFiles.length - 1].getPath()), job);
				if(reader == null)
					throw new IOException("matrix missing");
				TaggedIndex key = new TaggedIndex();
				MatrixObject value = new MatrixObject();
				while(reader.next(key, value))
				{
					matrixC = (SquareMatrix) value.getObject();
				}
				if(matrixC == null)
					throw new IOException("matrix missing");
				DistributedCache.releaseCache(cachedFiles[cachedFiles.length - 1], job);
			} catch (IOException e)
			{
				e.printStackTrace();
			}
		}
	}
	
	public static String runJob(int numMappers, int replication, String cInputDir,
			int updateType, String hwInputDir, String outputDir) throws IOException
	{
		String workingDirectory = outputDir + System.currentTimeMillis() + "-UpdateWHStep4/";

		JobConf job = new JobConf(UpdateWHStep4.class);
		job.setJobName("MatrixGNMFUpdateWHStep4");
		
		job.setInputFormat(SequenceFileInputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(hwInputDir));
		
		job.setOutputFormat(SequenceFileOutputFormat.class);
		FileOutputFormat.setOutputPath(job, new Path(workingDirectory));
		
		job.setNumMapTasks(numMappers);
		job.setMapperClass(UpdateWHStep4Mapper.class);
		job.setOutputKeyClass(TaggedIndex.class);
		job.setOutputValueClass(MatrixObject.class);
		
		job.setNumReduceTasks(0);

		DistributedCache.addCacheFile(new Path(cInputDir + "part-00000").toUri(), job);
		job.setInt("gnmf.updateType", updateType);
		
		JobClient.runJob(job);
		
		return workingDirectory;
	}
}
