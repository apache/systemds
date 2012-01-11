package gnmf.converter;

import gnmf.io.MatrixCell;
import gnmf.io.MatrixObject;
import gnmf.io.MatrixVector;
import gnmf.io.TaggedIndex;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

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
import org.apache.hadoop.mapred.TextInputFormat;

import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue.CellIndex;

public class DMLMatrixBlockToMatrixVectorsConverter
{
	protected enum Counters { VECTORS };
	public static final int CONVERTION_TYPE_ROW_WISE	= 0;
	public static final int CONVERTION_TYPE_COLUMN_WISE	= 1;
	
	static class ConverterMapper extends MapReduceBase
			implements Mapper<MatrixIndexes, MatrixBlock, IntWritable, MatrixCell>
	{
		private int convertionType;
		private int blkRowSize;
		private int blkColSize;
		
		@Override
		public void map(MatrixIndexes key, MatrixBlock value,
				OutputCollector<IntWritable, MatrixCell> out, Reporter reporter)
				throws IOException
		{
			if(value.isInSparseFormat())
			{
				value.sparseToDense();
			}
			
			int rowBase = (int) (key.getRowIndex() * blkRowSize);
			int columnBase = (int) (key.getColumnIndex() * blkColSize);
			double[] denseValues = value.getDenseArray();
			int rows = value.getNumRows();
			int columns = value.getNumColumns();
			for(int r = 0; r < rows; r++)
			{
				for(int c = 0; c < columns; c++)
				{
					if(convertionType == CONVERTION_TYPE_ROW_WISE)
					{
						out.collect(new IntWritable(rowBase + r),
								new MatrixCell((columnBase + c), denseValues[r * columns + c]));
					} else if(convertionType == CONVERTION_TYPE_COLUMN_WISE)
					{
						out.collect(new IntWritable(columnBase + c),
								new MatrixCell((rowBase + r), denseValues[r * columns + c]));
					}
				}
			}
		}
		
		@Override
		public void configure(JobConf job)
		{
			convertionType = job.getInt("gnmf.converter.convertionType", 0);
			blkRowSize = job.getInt("gnmf.converter.blkRowSize", 0);
			blkColSize = job.getInt("gnmf.converter.blkColSize", 0);
		}
	}
	
	static class ConverterReducer extends MapReduceBase
			implements Reducer<IntWritable, MatrixCell, TaggedIndex, MatrixObject>
	{
		@Override
		public void reduce(IntWritable key, Iterator<MatrixCell> values,
				OutputCollector<TaggedIndex, MatrixObject> out, Reporter reporter)
				throws IOException
		{
			HashMap<Integer, Double> currentValues = new HashMap<Integer, Double>();
			while(values.hasNext())
			{
				MatrixCell current = values.next();
				currentValues.put((int) current.getColumn(), current.getValue());
			}
			double[] valueArray = new double[currentValues.size()];
			for(int i = 0; i < currentValues.size(); i++)
			{
				valueArray[i] = currentValues.get(i);
			}
			out.collect(new TaggedIndex(key.get(), TaggedIndex.TYPE_VECTOR),
					new MatrixObject(new MatrixVector(valueArray)));
			reporter.incrCounter(Counters.VECTORS, 1);
		}
	}
	
	public static void convert(String inputDir, String outputDir, int convertionType,
			int blkRowSize, int blkColSize, int numMappers, int numReducers,
			int replication) throws IOException
	{
		Path workingDirectory = new Path(outputDir);
		
		JobConf job = new JobConf(DMLMatrixBlockToMatrixVectorsConverter.class);
		
		job.setJobName("DMLMatrixBlockToMatrixVectorsConverter");
		
		job.setMapperClass(DMLMatrixBlockToMatrixVectorsConverter.ConverterMapper.class);
		job.setReducerClass(DMLMatrixBlockToMatrixVectorsConverter.ConverterReducer.class);
		job.setNumMapTasks(numMappers);
		job.setNumReduceTasks(numReducers);
		
		job.setInputFormat(SequenceFileInputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(inputDir));
		
		job.setOutputFormat(SequenceFileOutputFormat.class);
		FileOutputFormat.setOutputPath(job, workingDirectory);
		
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(MatrixCell.class);
		job.setOutputKeyClass(TaggedIndex.class);
		job.setOutputValueClass(MatrixObject.class);
		
		job.setInt("dfs.replication", replication);
		
		if(FileSystem.get(job).exists(workingDirectory))
			FileSystem.get(job).delete(workingDirectory);
		
		job.setInt("gnmf.converter.convertionType", convertionType);
		job.setInt("gnmf.converter.blkRowSize", blkRowSize);
		job.setInt("gnmf.converter.blkColSize", blkColSize);
		
		JobClient.runJob(job);
	}
	
	public static void main(String[] args) throws IOException
	{
		if(args.length < 8)
		{
			System.out.println("missing parameters");
			System.out.println("expected parameters: inputDir outputDir convertionType " +
					"blkRowSize blkColSize numMappers numReducers replication");
			System.exit(1);
		}
		
		String inputDir		= args[0];
		String outputDir	= args[1];
		int convertionType	= Integer.parseInt(args[2]);
		int blkRowSize		= Integer.parseInt(args[3]);
		int blkColSize		= Integer.parseInt(args[4]);
		int numMappers		= Integer.parseInt(args[5]);
		int numReducers		= Integer.parseInt(args[6]);
		int replication		= Integer.parseInt(args[7]);
		
		DMLMatrixBlockToMatrixVectorsConverter.convert(inputDir, outputDir, convertionType,
				blkRowSize, blkColSize, numMappers, numReducers, replication);
	}
}
