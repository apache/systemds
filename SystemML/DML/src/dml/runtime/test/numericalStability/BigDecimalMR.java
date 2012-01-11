package dml.runtime.test.numericalStability;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.Counters.Group;


import dml.lops.LopProperties;
import dml.lops.runtime.RunMRJobs;
import dml.lops.runtime.RunMRJobs.ExecMode;
import dml.runtime.instructions.MRInstructions.PickByCountInstruction;
import dml.runtime.matrix.GMR;
import dml.runtime.matrix.JobReturn;
import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.NumItemsByEachReducerMetaData;
import dml.runtime.matrix.io.OutputInfo;
import dml.runtime.matrix.io.TaggedMatrixBlock;
import dml.runtime.matrix.io.TaggedMatrixCell;
import dml.runtime.matrix.mapred.GMRMapper;
import dml.runtime.matrix.mapred.GMRReducer;
import dml.runtime.matrix.mapred.MRJobConfiguration;
import dml.runtime.matrix.sort.PickFromCompactInputFormat;
import dml.runtime.util.MapReduceTool;
import dml.runtime.util.UtilFunctions;
import dml.utils.DMLRuntimeException;

public class BigDecimalMR {

	public static void runJob(String input, InputInfo inputInfo,  
			int brlen, int bclen, int replication,
			String output) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(BigDecimalMR.class);
		job.setJobName("BigDecimalMR");
		
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
		job.setMapperClass(BigDecimalMapper.class);
		job.setMapOutputKeyClass(NullWritable.class);
		job.setMapOutputValueClass(Text.class);
	
		//TODO: cannot set up combiner, because it will destroy the stable numerical algorithms 
		// for sum or for central moments 
		//set up combiner
	/*	if(numReducers!=0 && aggInstructionsInReducer!=null 
				&& !aggInstructionsInReducer.isEmpty())
			job.setCombinerClass(GMRCombiner.class);
	*/
		//configure reducer
		job.setReducerClass(BigDecimalReducer.class);
		//job.setReducerClass(PassThroughReducer.class);
		
		MapReduceTool.deleteFileIfExistOnHDFS(output);
		FileOutputFormat.setOutputPath(job, new Path(output));
		
		RunningJob runjob=JobClient.runJob(job);		
	}
	
	public static void main(String[] args) throws Exception {
		
		if(args.length<3)
		{
			System.out.println("BigDecimalMR <input> <output> <blockSize>");
			System.exit(-1);
		}
		int brlen=Integer.parseInt(args[2]);
		InputInfo inputinfo=InputInfo.TextCellInputInfo;
		if(brlen>1)
			inputinfo=InputInfo.BinaryBlockInputInfo;
		BigDecimalMR.runJob(args[0], inputinfo, brlen, 1, 
				1, args[1]);
	}
}
