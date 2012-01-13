package dml.runtime.test.numericalStability;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;

import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.mapred.MRJobConfiguration;
import dml.runtime.util.MapReduceTool;

public class NaiveUnivariate {

	public static void runJob(String input, InputInfo inputInfo,  
			int brlen, int bclen, int replication,
			String output) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(NaiveUnivariate.class);
		job.setJobName("NaiveUnivariate");
		
		//whether use block representation or cell representation
		MRJobConfiguration.setMatrixValueClass(job, false);
		
		//set up the input files and their format information
		MRJobConfiguration.setUpMultipleInputs(job, new byte[]{0}, new String[]{input}, new InputInfo[]{inputInfo}, 
				false, new int[]{brlen}, new int[]{bclen});
		
		//set up the block size
		MRJobConfiguration.setBlocksSizes(job, new byte[]{0}, new int[]{brlen}, new int[]{bclen});
		
		//set up the number of reducers
		job.setNumReduceTasks(1);
		
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);
		
		// configure mapper and the mapper output key value pairs
		job.setMapperClass(NaiveUnivariateMapper.class);
		job.setMapOutputKeyClass(DoubleWritable.class);
		job.setMapOutputValueClass(Text.class);
	
		//TODO: cannot set up combiner, because it will destroy the stable numerical algorithms 
		// for sum or for central moments 
		//set up combiner
	/*	if(numReducers!=0 && aggInstructionsInReducer!=null 
				&& !aggInstructionsInReducer.isEmpty())
			job.setCombinerClass(GMRCombiner.class);
	*/
		//configure reducer
		job.setReducerClass(NaiveUnivariateReducer.class);
		//job.setReducerClass(PassThroughReducer.class);
		
		MapReduceTool.deleteFileIfExistOnHDFS(output);
		FileOutputFormat.setOutputPath(job, new Path(output));
		
		RunningJob runjob=JobClient.runJob(job);		
	}
	
	public static void main(String[] args) throws Exception {
		
		if(args.length<3)
		{
			System.out.println("NaiveUnivariate <input> <output> <blockSize>");
			System.exit(-1);
		}
		int brlen=Integer.parseInt(args[2]);
		InputInfo inputinfo=InputInfo.TextCellInputInfo;
		if(brlen>1)
			inputinfo=InputInfo.BinaryBlockInputInfo;
		NaiveUnivariate.runJob(args[0], inputinfo, brlen, 1, 
				1, args[1]);
	}
}
