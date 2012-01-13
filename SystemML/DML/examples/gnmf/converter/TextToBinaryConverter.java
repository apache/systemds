package gnmf.converter;

import gnmf.io.Matrix;
import gnmf.io.MatrixCell;
import gnmf.io.MatrixFormats;
import gnmf.io.MatrixObject;
import gnmf.io.MatrixVector;
import gnmf.io.TaggedIndex;

import java.io.IOException;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;


/**
 * Contains classes which can be used to convert text based objects into classes.
 * 
 * @author schnetter
 */
public class TextToBinaryConverter
{
	static class SparseMatrixToMatrixCellConverter extends MapReduceBase
			implements Mapper<LongWritable, Text, TaggedIndex, MatrixObject>
	{
		@Override
		public void map(LongWritable line, Text rcvString,
				OutputCollector<TaggedIndex, MatrixObject> out, Reporter reporter)
				throws IOException
		{
			String[] rcv = rcvString.toString().split(" ");
			long row		= Long.parseLong(rcv[0]);
			long column		= Long.parseLong(rcv[1]);
			double value	= Double.parseDouble(rcv[2]);
			out.collect(new TaggedIndex(row, TaggedIndex.TYPE_CELL),
					new MatrixObject(new MatrixCell(column, value)));
		}
	}
	
	public static String convertSparseMatrixToMatrixCell(String inputDir,
			String outputDir, int numMappers, int replication) throws IOException
	{
		String workingDirectory = outputDir + System.currentTimeMillis() + "-MatrixCells/";
		
		JobConf job = new JobConf(TextToBinaryConverter.class);
		
		job.setJobName("SparseMatrixToMatrixCellConverter");
		
		job.setMapperClass(SparseMatrixToMatrixCellConverter.class);
		job.setNumMapTasks(numMappers);
		job.setNumReduceTasks(0);
		
		job.setInputFormat(TextInputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(inputDir));
		
		job.setOutputFormat(SequenceFileOutputFormat.class);
		FileOutputFormat.setOutputPath(job, new Path(workingDirectory));
		
		job.setOutputKeyClass(TaggedIndex.class);
		job.setOutputValueClass(MatrixObject.class);
		
		job.setInt("dfs.replication", replication);
		
		JobClient.runJob(job);
		
		return workingDirectory;
	}
	
	
	static class VectorToMatrixVectorConverter extends MapReduceBase
			implements Mapper<LongWritable, Text, TaggedIndex, MatrixObject>
	{
		@Override
		public void map(LongWritable line, Text vectorString,
				OutputCollector<TaggedIndex, MatrixObject> out, Reporter reporter)
				throws IOException
		{
			String[] vector = vectorString.toString().split(" ");
			long row		= Long.parseLong(vector[0]);
			double[] values	= new double[vector.length - 1];
			for(int i = 1; i < vector.length; i++)
			{
				values[i - 1] = Double.parseDouble(vector[i]);
			}
			out.collect(new TaggedIndex(row, TaggedIndex.TYPE_VECTOR),
					new MatrixObject(new MatrixVector(values)));
		}
	}
	
	public static String convertVectorToMatrixVector(String inputDir,
			String outputDir, int numMappers, int replication) throws IOException
	{
		String workingDirectory = outputDir + System.currentTimeMillis() + "-MatrixVector/";
		
		JobConf job = new JobConf(TextToBinaryConverter.class);
		
		job.setJobName("VectorToMatrixVectorConverter");
		
		job.setMapperClass(VectorToMatrixVectorConverter.class);
		job.setNumMapTasks(numMappers);
		job.setNumReduceTasks(0);
		
		job.setInputFormat(TextInputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(inputDir));
		
		job.setOutputFormat(SequenceFileOutputFormat.class);
		FileOutputFormat.setOutputPath(job, new Path(workingDirectory));
		
		job.setOutputKeyClass(TaggedIndex.class);
		job.setOutputValueClass(MatrixObject.class);
		
		job.setInt("dfs.replication", replication);
		
		JobClient.runJob(job);
		
		return workingDirectory;
	}
}
