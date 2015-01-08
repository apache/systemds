package gnmf.converter;

import gnmf.io.MatrixCell;
import gnmf.io.MatrixObject;
import gnmf.io.TaggedIndex;

import java.io.IOException;
import java.util.HashMap;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
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

import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue.CellIndex;

public class DMLMatrixBlockToMatrixCellsConverter
{
	protected enum Counters { CELLS };
	
	
	static class ConverterMapper extends MapReduceBase
			implements Mapper<MatrixIndexes, MatrixBlock, TaggedIndex, MatrixObject>
	{
		private int blkRowSize;
		private int blkColSize;
		
		@Override
		public void map(MatrixIndexes key, MatrixBlock value,
				OutputCollector<TaggedIndex, MatrixObject> out, Reporter reporter)
				throws IOException
		{
			int rowBase = (int) (key.getRowIndex() * blkRowSize);
			int columnBase = (int) (key.getColumnIndex() * blkColSize);
			if(value.isInSparseFormat())
			{
				HashMap<CellIndex, Double> sparseMap = value.getSparseMap();
				for(CellIndex current : sparseMap.keySet())
				{
					out.collect(new TaggedIndex((rowBase + current.row),
								TaggedIndex.TYPE_CELL),
							new MatrixObject(new MatrixCell((columnBase + current.column),
									sparseMap.get(current).doubleValue())));
					reporter.incrCounter(Counters.CELLS, 1);
				}
			} else
			{
				double[] denseValues = value.getDenseArray();
				int rows = value.getNumRows();
				int columns = value.getNumColumns();
				for(int r = 0; r < rows; r++)
				{
					for(int c = 0; c < columns; c++)
					{
						out.collect(new TaggedIndex((rowBase + r), TaggedIndex.TYPE_CELL),
								new MatrixObject(new MatrixCell((columnBase + c),
										denseValues[r * columns + c])));
						reporter.incrCounter(Counters.CELLS, 1);
					}
				}
			}
		}
		
		@Override
		public void configure(JobConf job)
		{
			blkRowSize = job.getInt("gnmf.converter.blkRowSize", 0);
			blkColSize = job.getInt("gnmf.converter.blkColSize", 0);
		}
	}
	
	public static void convert(String inputDir, String outputDir, int blkRowSize,
			int blkColSize, int numMappers, int replication) throws IOException
	{
		Path workingDirectory = new Path(outputDir);
		
		JobConf job = new JobConf(DMLMatrixBlockToMatrixCellsConverter.class);
		
		job.setJobName("DMLMatrixBlockToMatrixCellsConverter");
		
		job.setMapperClass(DMLMatrixBlockToMatrixCellsConverter.ConverterMapper.class);
		job.setNumMapTasks(numMappers);
		job.setNumReduceTasks(0);
		
		job.setInputFormat(SequenceFileInputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(inputDir));
		
		job.setOutputFormat(SequenceFileOutputFormat.class);
		FileOutputFormat.setOutputPath(job, workingDirectory);
		
		job.setOutputKeyClass(TaggedIndex.class);
		job.setOutputValueClass(MatrixObject.class);
		
		job.setInt("dfs.replication", replication);
		
		if(FileSystem.get(job).exists(workingDirectory))
			FileSystem.get(job).delete(workingDirectory);
		
		job.setInt("gnmf.converter.blkRowSize", blkRowSize);
		job.setInt("gnmf.converter.blkColSize", blkColSize);
		
		JobClient.runJob(job);
	}
	
	public static void main(String[] args) throws IOException
	{
		if(args.length < 6)
		{
			System.out.println("missing parameters");
			System.out.println("expected parameters: inputDir outputDir blkRowSize blkColSize " +
					"numMappers replication");
			return;
		}
		
		String inputDir		= args[0];
		String outputDir	= args[1];
		int blkRowSize		= Integer.parseInt(args[2]);
		int blkColSize		= Integer.parseInt(args[3]);
		int numMappers		= Integer.parseInt(args[4]);
		int replication		= Integer.parseInt(args[5]);
		
		DMLMatrixBlockToMatrixCellsConverter.convert(inputDir, outputDir, blkRowSize,
				blkColSize, numMappers, replication);
	}
}
