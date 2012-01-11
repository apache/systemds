package dml.runtime.test.numericalStability;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.SequenceFileInputFormat;

import dml.runtime.matrix.CombineMR;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;
import dml.runtime.util.MapReduceTool;

public class BigDecimalCovarianceMR {

	public static void runJob(String input,  int replication, String output) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(BigDecimalCovarianceMR.class);
		job.setJobName("BigDecimalCovarianceMR");
		
		FileInputFormat.addInputPath(job, new Path(input));
		job.setInputFormat(SequenceFileInputFormat.class);
		//set up the number of reducers
		job.setNumReduceTasks(1);
		
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);
				
		// configure mapper and the mapper output key value pairs
		job.setMapperClass(BigDecimalCovarianceMapper.class);
		
		job.setMapOutputKeyClass(NullWritable.class);
		job.setMapOutputValueClass(Text.class);
		
		//configure reducer
		job.setReducerClass(BigDecimalCovarianceReducer.class);
		MapReduceTool.deleteFileIfExistOnHDFS(output);
		FileOutputFormat.setOutputPath(job, new Path(output));
		
		RunningJob runjob=JobClient.runJob(job);		
	}
	
	public static void main(String[] args) throws Exception {
		if(args.length<6)
		{
			System.out.println("BigDecimalCovarianceMR <input1> <input2> <num Rows> <# reducers> <output> <blocksize>");
			System.exit(-1);
		}
		
		boolean inBlockRepresentation=false;
		String V=args[0];
		String U=args[1];
		String[] inputs=new String[]{V, U};
		int npb=Integer.parseInt(args[5]);
		InputInfo[] inputInfos=new InputInfo[]{InputInfo.TextCellInputInfo, InputInfo.TextCellInputInfo};
		if(npb>1)
		{
			inputInfos=new InputInfo[]{InputInfo.BinaryBlockInputInfo, InputInfo.BinaryBlockInputInfo};
			inBlockRepresentation=true;
		}
		long r=Long.parseLong(args[2]);
		long[] rlens=new long[]{r, r};
		long[] clens=new long[]{1, 1};
		int[] brlens=new int[]{npb, npb};
		int[] bclens=new int[]{1, 1};
		String combineInstructions="combinebinary\u00b0false\u00b7BOOLEAN\u00b00\u00b7DOUBLE\u00b01\u00b7DOUBLE\u00b02\u00b7DOUBLE";
		int numReducers=Integer.parseInt(args[3]);
		int replication=1;
		byte[] resultIndexes=new byte[]{2};
		String UV="UV";
		String[] outputs=new String[]{UV};
		OutputInfo[] outputInfos=new OutputInfo[]{OutputInfo.WeightedPairOutputInfo};
		CombineMR.runJob(inBlockRepresentation, inputs, inputInfos, rlens, clens, brlens, bclens, 
				combineInstructions, numReducers, replication, resultIndexes, outputs, outputInfos);
		
		BigDecimalCovarianceMR.runJob(UV, replication, args[4]);
	}
}
