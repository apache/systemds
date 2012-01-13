package dml.runtime.test;

import java.io.IOException;

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

import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixCell;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.mapred.MRJobConfiguration;
import dml.runtime.util.MapReduceTool;

public class BlockToCell {

	static class InnerMapper extends MapReduceBase implements
			Mapper<MatrixIndexes, MatrixBlock, MatrixIndexes, MatrixCell> {
		int brlen;
		int bclen;
		MatrixIndexes indexBuffer = new MatrixIndexes();
		MatrixCell cellBuffer = new MatrixCell();

		@Override
		public void map(MatrixIndexes indexes, MatrixBlock block, OutputCollector<MatrixIndexes, MatrixCell> out,
				Reporter report) throws IOException {

			for (int i = 0; i < block.getNumRows(); i++)
				for (int j = 0; j < block.getNumColumns(); j++) {
					double value = block.getValue(i, j);
					if (value != 0) {
						indexBuffer.setIndexes(indexes.getRowIndex() * brlen + i, indexes.getColumnIndex() * bclen + j);
						cellBuffer.setValue(value);
						out.collect(indexBuffer, cellBuffer);
						// System.out.println(indexBuffer+": "+cellBuffer);
					}
				}

		}

		public void configure(JobConf job) {
			brlen = MRJobConfiguration.getNumRowsPerBlock(job, (byte) 0);
			bclen = MRJobConfiguration.getNumColumnsPerBlock(job, (byte) 0);
		}
	}

	public static void runJob(String input, String output, int replication, int bnr, int bnc) throws Exception {
		JobConf job;
		job = new JobConf(BlockToCell.class);
		job.setJobName("BlockToCell");
		FileInputFormat.addInputPath(job, new Path(input));
		job.setInputFormat(SequenceFileInputFormat.class);

		// configure reducer
		job.setInt("dfs.replication", replication);
		MRJobConfiguration.setBlockSize(job, (byte) 0, bnr, bnc);

		// configure output
		Path outPath = new Path(output);
		FileOutputFormat.setOutputPath(job, new Path(output));
		FileOutputFormat.setOutputPath(job, outPath);
		MapReduceTool.deleteFileIfExistOnHDFS(outPath, job);

		// configure mapper
		job.setMapperClass(InnerMapper.class);
		job.setMapOutputKeyClass(MatrixIndexes.class);
		job.setMapOutputValueClass(MatrixCell.class);

		// configure reducer
		job.setNumReduceTasks(0);
		job.setOutputKeyClass(MatrixIndexes.class);
		job.setOutputValueClass(MatrixCell.class);
		job.setOutputFormat(SequenceFileOutputFormat.class);

		JobClient.runJob(job);
	}

	public static void main(String[] args) throws Exception {
		if (args.length < 5) {
			System.err.println("expect 5 paramers: input, output, replication, bnr, bnc");
			System.exit(-1);
		}
		String input = args[0];
		String outDir = args[1];
		int replication = Integer.parseInt(args[2]);
		int bnr = Integer.parseInt(args[3]);
		int bnc = Integer.parseInt(args[4]);

		BlockToCell.runJob(input, outDir, replication, bnr, bnc);
	}
}
