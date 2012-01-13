package dml.runtime.test;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
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
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;

import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.util.MapReduceTool;

public class GenerateTestMatrixBlockVerySparse {

	protected static enum Counters {NON_ZEROS};
//	protected static final Log LOG = LogFactory.getLog(GenerateTestMatrix.class);
	public static int maxRandom=5000;
	static class SequenceOutMapper extends MapReduceBase implements Mapper<LongWritable, Text, 
	MatrixIndexes, MatrixBlock>
	{
		private MatrixIndexes indexes=new MatrixIndexes();
		private MatrixBlock block=new MatrixBlock();
		private Random random;
		@Override
		public void map(LongWritable key, Text value,
				OutputCollector<MatrixIndexes, MatrixBlock> out,
				Reporter report) throws IOException {
			String[] strs=value.toString().split(",");
			long i=Long.parseLong(strs[0]);
			long j=Long.parseLong(strs[1]);
			int blockRlen=Integer.parseInt(strs[2]);
			int blockClen=Integer.parseInt(strs[3]);
			long seed=Long.parseLong(strs[4]);
			random=new Random(seed);
			double sparcity=Double.parseDouble(strs[5]);
			indexes.setIndexes(i, j);
			
			if(sparcity>0.333)
				block.reset(blockRlen, blockClen, false);
			else
				block.reset(blockRlen, blockClen, true);
			int limit=blockRlen*blockClen;
			int num=(int) (limit*sparcity);
			double prob=limit*sparcity-num;
			if(random.nextDouble()<prob)
				num++;
			for(int n=0; n<num; n++)
			{
				int index=random.nextInt(limit);
				int bi=index/blockClen;
				int bj=index%blockClen;
				if(block.getValue(bi, bj)==0)
					report.incrCounter(Counters.NON_ZEROS, 1);
				block.setValue(bi, bj, (double)random.nextInt(maxRandom)/(double)maxRandom);
			}
			
			//if(block.getNonZeros()>0)
		//	{
				out.collect(indexes, block);
			//	LOG.info("output: \n"+indexes+"\n"+block);
			//	LOG.info("nonZeros = "+ block.getNonZeros());
		//	}
		}
	}
	
	public static void generateJobFile(JobConf job, Path jobFile, long numRows, 
			long numColumns, int blockRowSize, int blockColumnsSize, long seed, 
			double sparcity) throws IOException {
		FileSystem fs = FileSystem.get(job);
		FSDataOutputStream fsout = fs.create(jobFile);
		PrintWriter out = new PrintWriter(fsout);
		
		for(long i=0; i<numRows; i+=blockRowSize)
		{
			long realBlockRlen=Math.min(blockRowSize, numRows-i);
			for(long j=0; j<numColumns; j+=blockColumnsSize)
			{
				long realBlockClen=Math.min(blockColumnsSize, numColumns-j);
				out.println(i/blockRowSize + "," + j/blockColumnsSize + "," + realBlockRlen+","+ realBlockClen +","+(seed+i*numRows+j)+","+sparcity);
			}
		}
		out.close();
	}
	
	public static JobConf runJob(int numMappers, long numRows, long numColumns, String outDir, 
			int replication, int blockRowSize, int blockColumnsSize, long seed, double sparcity) 
	throws IOException
	{
		Path jobFile = new Path( "GenerateTestMatrix-seeds-"
				+ Integer.toString(new Random().nextInt(Integer.MAX_VALUE)));
		Path outputDir=new Path(outDir);
		// create job
		JobConf job = new JobConf(GenerateTestMatrixBlockVerySparse.class);
		try { 
			job.setJobName("GenerateTestMatrix");
			
			// create job file
			generateJobFile(job, jobFile, numRows, numColumns, blockRowSize, blockColumnsSize, seed, sparcity);
			
			// configure input
			job.setInputFormat(TextInputFormat.class);
			FileInputFormat.setInputPaths(job, jobFile);
			
			// configure mappers
			job.setNumMapTasks(numMappers);
			job.setMapperClass(SequenceOutMapper.class);
			job.setMapOutputKeyClass(MatrixIndexes.class);
			job.setMapOutputValueClass(MatrixBlock.class);
			job.setOutputFormat(SequenceFileOutputFormat.class);
			
			job.setOutputKeyClass(MatrixIndexes.class);
			job.setOutputValueClass(MatrixBlock.class);
			// configure reducers
			job.setNumReduceTasks(0);
		//	job.setReducerClass(IdentityReducer.class);
			
			// configure output
			FileOutputFormat.setOutputPath(job, outputDir);
			MapReduceTool.deleteFileIfExistOnHDFS(outputDir, job);
			job.setInt("dfs.replication", replication);
			
			JobClient.runJob(job);
		} finally {
			MapReduceTool.deleteFileIfExistOnHDFS(jobFile, job);
		}
		return job;
	}
	
	public static void main(String[] args) throws Exception {
		if(args.length<9)
		{
			System.err.println("expect 9 paramers: numMappers numRows numColumns " +
					"outputDir replication blockRowSize, blockColumnsSize, seed sparcity");
			System.exit(-1);
		}
		
		int numMappers=Integer.parseInt(args[0]);
		long numRows=Long.parseLong(args[1]);
		long numColumns=Long.parseLong(args[2]);
		String outDir=args[3];
		int replication=Integer.parseInt(args[4]);
		int blockRowSize=Integer.parseInt(args[5]);
		int blockColumnsSize=Integer.parseInt(args[6]);
		long seed=Long.parseLong(args[7]);
		double sparcity=Double.parseDouble(args[8]);
		JobConf job=GenerateTestMatrixBlockVerySparse.runJob(numMappers, numRows, numColumns, outDir, 
				replication, blockRowSize, blockColumnsSize, seed, sparcity);
	}
}
