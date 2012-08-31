package com.ibm.bi.dml.runtime.matrix;

import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;

import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.lops.runtime.RunMRJobs;
import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObjectNew;
import com.ibm.bi.dml.runtime.matrix.io.CM_N_COVCell;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.CMCOVMRMapper;
import com.ibm.bi.dml.runtime.matrix.mapred.CMCOVMRReducer;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;


public class CMCOVMR {
	
	public static JobReturn runJob(String[] inputVars, MatrixObjectNew[] inputMatrices, 
			String instructionsInMapper, String cmNcomInstructions, 
			String[] outputVars, MatrixObjectNew[] outputMatrices, byte[] resultIndexes,
			int numReducers, int replication) throws Exception {
		String[] inputs = new String[inputMatrices.length];
		InputInfo[] inputInfos = new InputInfo[inputMatrices.length];
		long[] rlens = new long[inputMatrices.length];
		long[] clens = new long[inputMatrices.length];
		int[] brlens = new int[inputMatrices.length];
		int[] bclens = new int[inputMatrices.length];
		
		String[] outputs = new String[outputVars.length];
		OutputInfo[] outputInfos = new OutputInfo[outputVars.length];
		
		GMR.populateInputs(inputVars, inputMatrices, inputs, inputInfos, rlens, clens, brlens, bclens);
		GMR.populateOutputs(outputVars, outputMatrices, outputs, outputInfos);
		
		return runJob(inputs, inputInfos, rlens, clens, brlens, bclens, 
				instructionsInMapper, cmNcomInstructions,
				numReducers, replication, resultIndexes, outputs, outputInfos);
	}
	
	public static JobReturn runJob(String[] inputs, InputInfo[] inputInfos, long[] rlens, long[] clens, 
			int[] brlens, int[] bclens, String instructionsInMapper, String cmNcomInstructions, 
			int numReducers, int replication, byte[] resultIndexes,	String[] outputs, OutputInfo[] outputInfos) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(CMCOVMR.class);
		job.setJobName("CM-COV-MR");
		
		//whether use block representation or cell representation
		MRJobConfiguration.setMatrixValueClassForCM_N_COM(job, true);
	
		//added for handling recordreader instruction
		String[] realinputs=inputs;
		InputInfo[] realinputInfos=inputInfos;
		long[] realrlens=rlens;
		long[] realclens=clens;
		int[] realbrlens=brlens;
		int[] realbclens=bclens;
		byte[] realIndexes=new byte[inputs.length];
		for(byte b=0; b<realIndexes.length; b++)
			realIndexes[b]=b;
		
		//set up the input files and their format information
		MRJobConfiguration.setUpMultipleInputs(job, realIndexes, realinputs, realinputInfos, 
				false, realbrlens, realbclens, true, true);
		
		//set up the dimensions of input matrices
		MRJobConfiguration.setMatricesDimensions(job, realIndexes, realrlens, realclens);
		
		//set up the block size
		MRJobConfiguration.setBlocksSizes(job, realIndexes, realbrlens, realbclens);
		
		//set up unary instructions that will perform in the mapper
		MRJobConfiguration.setInstructionsInMapper(job, instructionsInMapper);
		
		//set up the aggregate instructions that will happen in the combiner and reducer
		MRJobConfiguration.setCM_N_COMInstructions(job, cmNcomInstructions);
		
		//set up the number of reducers
		job.setNumReduceTasks(numReducers);
		
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);
		
		//set up what matrices are needed to pass from the mapper to reducer
		MRJobConfiguration.setUpOutputIndexesForMapper(job, realIndexes, instructionsInMapper, null, 
				cmNcomInstructions, resultIndexes);
		
		//set up the multiple output files, and their format information
		MRJobConfiguration.setUpMultipleOutputs(job, resultIndexes, new byte[resultIndexes.length], outputs, outputInfos, false);
		
		// configure mapper and the mapper output key value pairs
		job.setMapperClass(CMCOVMRMapper.class);
		
		job.setMapOutputKeyClass(TaggedFirstSecondIndexes.class);
		job.setMapOutputValueClass(CM_N_COVCell.class);
		job.setOutputKeyComparatorClass(TaggedFirstSecondIndexes.Comparator.class);
		job.setPartitionerClass(TaggedFirstSecondIndexes.TagPartitioner.class);
		
		//configure reducer
		job.setReducerClass(CMCOVMRReducer.class);
		//job.setReducerClass(PassThroughReducer.class);
		
		MatrixCharacteristics[] stats=MRJobConfiguration.computeMatrixCharacteristics(job, realIndexes, 
				instructionsInMapper, null, null, cmNcomInstructions, resultIndexes);
		
		
		// By default, the job executes in "cluster" mode.
		// Determine if we can optimize and run it in "local" mode.
		MatrixCharacteristics[] inputStats = new MatrixCharacteristics[inputs.length];
		for ( int i=0; i < inputs.length; i++ ) {
			inputStats[i] = new MatrixCharacteristics(rlens[i], clens[i], brlens[i], bclens[i]);
		}
		
		ExecMode mode = RunMRJobs.getExecMode(JobType.CM_COV, inputStats); 
		if ( mode == ExecMode.LOCAL ) {
			job.set("mapred.job.tracker", "local");
			MRJobConfiguration.setStagingDir( job );
		}
		
		//set unique working dir
		MRJobConfiguration.setUniqueWorkingDir(job, mode);
		
		
		RunningJob runjob=JobClient.runJob(job);
		
		return new JobReturn(stats, outputInfos, runjob.isSuccessful());
	}

}
