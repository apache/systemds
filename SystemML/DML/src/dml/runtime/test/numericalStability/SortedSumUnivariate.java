package dml.runtime.test.numericalStability;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import dml.runtime.instructions.Instruction;
import dml.runtime.matrix.JobReturn;
import dml.runtime.matrix.SortMR;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.NumItemsByEachReducerMetaData;
import dml.runtime.matrix.io.OutputInfo;
import dml.runtime.matrix.mapred.MRJobConfiguration;
import dml.runtime.matrix.sort.CompactOutputFormat;
import dml.runtime.matrix.sort.PickFromCompactInputFormat;
import dml.runtime.util.MapReduceTool;

public class SortedSumUnivariate {

	public static void runJob(String input, InputInfo inputInfo,  
			int brlen, int bclen, int replication,
			String output) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(SortedSumUnivariate.class);
		job.setJobName("SortedSumUnivariate");
		
		//whether use block representation or cell representation
		MRJobConfiguration.setMatrixValueClass(job, false);
		
		//set up the input files and their format information
		PickFromCompactInputFormat.setKeyValueClasses(job, (Class<? extends WritableComparable>) inputInfo.inputKeyClass, inputInfo.inputValueClass);
	    PickFromCompactInputFormat.setPickRecordsInEachPartFile(job, (NumItemsByEachReducerMetaData) inputInfo.metadata, 0, 1);
		// job.setInputFormat(PickFromCompactInputFormat.class);
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
		
		if(args.length<5)
		{
			System.err.println("SortedSumUnivariate <input> <output> <numRows> <blockSize> <#reducers>");
			System.exit(-1);
		}
		int numrows = Integer.parseInt(args[2]);
		int brlen=Integer.parseInt(args[3]);
		
		Logger mapredLogger = Logger.getLogger("org.apache.hadoop.mapred");
		mapredLogger.setLevel(Level.FATAL);
		
		// Sort the input file
		InputInfo inputinfo=InputInfo.TextCellInputInfo;
		if(brlen>1)
			inputinfo=InputInfo.BinaryBlockInputInfo;

	    OutputInfo outputInfo=new OutputInfo(CompactOutputFormat.class, DoubleWritable.class, IntWritable.class);
	    String combineInstr 
	    	= "combineunary" 
			  + Instruction.OPERAND_DELIM + "0" + Instruction.VALUETYPE_PREFIX + "DOUBLE" 
			  + Instruction.OPERAND_DELIM + "1" + Instruction.VALUETYPE_PREFIX + "DOUBLE"; 
	    
	    String sortedfile = "SortedDataFile";
	    
	    JobReturn ret=SortMR.runJob(args[0], inputinfo, numrows, 1, brlen, 1, combineInstr, 
	    		Integer.parseInt(args[4]), 1, (byte) 0, sortedfile, outputInfo, false);
		
	    // Compute the sum using the sorted list
	    inputinfo.inputFormatClass=PickFromCompactInputFormat.class;
	    inputinfo.inputKeyClass=DoubleWritable.class;
	    inputinfo.inputValueClass=IntWritable.class;
	    inputinfo.metadata=ret.getMetaData(0);
	    SortedSumUnivariate.runJob(sortedfile, inputinfo, 1, 1, 1, args[1]);
	}
}
