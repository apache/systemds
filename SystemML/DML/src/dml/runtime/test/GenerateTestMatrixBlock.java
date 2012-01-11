package dml.runtime.test;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
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
import org.apache.hadoop.mapred.lib.IdentityReducer;

import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.util.MapReduceTool;

import cern.jet.random.Beta;
import cern.jet.random.Exponential;
import cern.jet.random.engine.RandomEngine;

public class GenerateTestMatrixBlock{

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
			
			if(sparcity>0.25)
				block.reset(blockRlen, blockClen, false);
			else
				block.reset(blockRlen, blockClen, true);
			
			for(int bi=0; bi<blockRlen; bi++)
				for(int bj=0; bj<blockClen; bj++)
				{
					if(random.nextDouble()>sparcity)
						continue;
					block.setValue(bi, bj, (double)random.nextInt(maxRandom)/(double)maxRandom);
				}
		//	if(block.getNonZeros()>0)
		//	{
				out.collect(indexes, block);
				report.incrCounter(Counters.NON_ZEROS, block.getNonZeros());
			//	System.out.println("output: "+indexes);
			//	LOG.info("nonZeros = "+ block.getNonZeros());
			//}
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
				out.println((i/blockRowSize+1) + "," + (j/blockColumnsSize+1) + "," + realBlockRlen+","+ realBlockClen +","+(seed+i*numRows+j)+","+sparcity);
			}
		}
		out.close();
	}
	
	public static void generateJobFileNormal(JobConf job, Path jobFile, long numRows, 
			long numColumns, int blockRowSize, int blockColumnsSize, long seed, 
			double sparcity)throws IOException
	{
		FileSystem fs = FileSystem.get(job);
		FSDataOutputStream fsout = fs.create(jobFile);
		PrintWriter out = new PrintWriter(fsout);
		double alpha=1.0/(1-sparcity);
		double beta=1.0/sparcity;
		
		//double lambda=1.5;//1.0/sparcity/blockRowSize/blockColumnsSize;
		//Exponential exp=new Exponential(lambda, RandomEngine.makeDefault());
		Beta dist=new Beta(alpha, beta, RandomEngine.makeDefault());
		
		for(long i=0; i<numRows; i+=blockRowSize)
		{
			long realBlockRlen=Math.min(blockRowSize, numRows-i);
			for(long j=0; j<numColumns; j+=blockColumnsSize)
			{
				double realsparsity=dist.nextDouble();
				long realBlockClen=Math.min(blockColumnsSize, numColumns-j);
				out.println((i/blockRowSize+1) + "," + (j/blockColumnsSize+1) + "," + realBlockRlen+","+ realBlockRlen +","+(seed+i*numRows+j)+","+realsparsity);
			}
		}
		out.close();
	}
	
	public static JobConf runJob(int numMappers, long numRows, long numColumns, String outDir, 
			int replication, int blockRowSize, int blockColumnsSize, long seed, double sparcity, boolean uniform) 
	throws IOException
	{
		Path jobFile = new Path( "GenerateTestMatrix-seeds-"
				+ Integer.toString(new Random().nextInt(Integer.MAX_VALUE)));
		Path outputDir=new Path(outDir);
		// create job
		JobConf job = new JobConf(GenerateTestMatrixBlock.class);
		try { 
			job.setJobName("GenerateTestMatrix");
			
			// create job file
			if(uniform)
				generateJobFile(job, jobFile, numRows, numColumns, blockRowSize, blockColumnsSize, seed, sparcity);
			else
				generateJobFileNormal(job, jobFile, numRows, numColumns, blockRowSize, blockColumnsSize, seed, sparcity);
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
		if(args.length<10)
		{
			System.err.println("expect 9 paramers: numMappers numRows numColumns " +
					"outputDir replication blockRowSize, blockColumnsSize, seed, sparcity, uniform distribution");
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
		boolean uniform=Boolean.parseBoolean(args[9]);
		JobConf job=GenerateTestMatrixBlock.runJob(numMappers, numRows, numColumns, outDir, 
				replication, blockRowSize, blockColumnsSize, seed, sparcity, uniform);
		/*
		FileSystem fs=FileSystem.getLocal(job);
		Path toread=new Path("something");//new Path(outDir+"/part-00000");//new Path("A.binary/0-r-00000");
		SequenceFile.Writer writer=new SequenceFile.Writer(fs, job, toread, MatrixIndexes.class, MatrixBlock.class); 
		MatrixIndexes indexes=new MatrixIndexes(0, 1);
		MatrixBlock block=new MatrixBlock(2, 2, true);
		block.setValue(1, 0, 2.0);
		
		for(int i=0; i<10; i++)
			writer.append(indexes, block);
		writer.close();
		
		SequenceFile.Reader reader=new SequenceFile.Reader(fs, toread, job);
		
		while(reader.next(indexes, block))
		{
			System.out.println("read one record:");
			System.out.println(indexes);
			System.out.println(block);
		} 
		if(reader!=null)
			reader.close();*/
	}
}
