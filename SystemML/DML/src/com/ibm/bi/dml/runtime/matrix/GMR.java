package com.ibm.bi.dml.runtime.matrix;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.Counters.Group;

import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.lops.runtime.RunMRJobs;
import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObjectNew;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.PickByCountInstruction;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.NumItemsByEachReducerMetaData;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.TaggedMatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.TaggedMatrixPackedCell;
import com.ibm.bi.dml.runtime.matrix.mapred.GMRCombiner;
import com.ibm.bi.dml.runtime.matrix.mapred.GMRMapper;
import com.ibm.bi.dml.runtime.matrix.mapred.GMRReducer;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.matrix.sort.PickFromCompactInputFormat;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

 
public class GMR{
	/*
	 * inBlockRepresentation: indicate whether to use block representation or cell representation
	 * inputs: input matrices, the inputs are indexed by 0, 1, 2, .. based on the position in this string
	 * inputInfos: the input format information for the input matrices
	 * rlen: the number of rows for each matrix
	 * clen: the number of columns for each matrix
	 * brlen: the number of rows per block
	 * bclen: the number of columns per block
	 * instructionsInMapper: in Mapper, the set of unary operations that need to be performed on each input matrix
	 * aggInstructionsInReducer: in Reducer, right after sorting, the set of aggreagte operations that need 
	 * 							to be performed on each input matrix, 
	 * otherInstructionsInReducer: the mixed operations that need to be performed on matrices after the aggregate operations
	 * numReducers: the number of reducers
	 * replication: the replication factor for the output
	 * resulltIndexes: the indexes of the result matrices that needs to be outputted.
	 * outputs: the names for the output directories, one for each result index
	 * outputInfos: output format information for the output matrices
	 */
	
	/**
	 * Given an array of input variable names and corresponding MatrixObjects (from symbol table), 
	 * this function updates the remaining parameters (rlens, clens, ... ) with the information required 
	 * to spawn the MapReduce job. 
	 */
	static void populateInputs(String []inputVars, MatrixObjectNew[] inputMatrices, String[] inputs, InputInfo[] inputInfos, long[] rlens, long[] clens, int[] brlens, int []bclens) {
		// Since inputVars can potentially contain scalar variables,
		// the loop is made over inputMatrices and not inputVars
		for ( int i=0; i < inputMatrices.length; i++ ) {
			inputs[i] = inputMatrices[i].getFileName();
			MatrixCharacteristics mc = ((MatrixDimensionsMetaData) inputMatrices[i].getMetaData()).getMatrixCharacteristics();
			rlens[i] = mc.get_rows();
			clens[i] = mc.get_cols();
			brlens[i] = mc.get_rows_per_block();
			bclens[i] = mc.get_cols_per_block();
			if ( inputMatrices[i].getMetaData() instanceof MatrixFormatMetaData ) {
				inputInfos[i] = ((MatrixFormatMetaData) inputMatrices[i].getMetaData()).getInputInfo();
			}
			else if (inputMatrices[i].getMetaData() instanceof NumItemsByEachReducerMetaData ) {
				inputInfos[i] = InputInfo.InputInfoForSortOutput;
				inputInfos[i].metadata = inputMatrices[i].getMetaData();
			}
		}
	}
	
	static void populateOutputs(String[] outputVars, MatrixObjectNew[] outputMatrices, String[] outputs, OutputInfo[] outputInfos) {
		for(int i=0; i < outputVars.length; i++) {
			outputs[i] = outputMatrices[i].getFileName();
			MatrixFormatMetaData md = (MatrixFormatMetaData) outputMatrices[i].getMetaData();
			outputInfos[i] = md.getOutputInfo();
		}
	}
	
	public static JobReturn runJob(String[] inputVars, MatrixObjectNew[] inputMatrices, 
			String recordReaderInstruction, String instructionsInMapper, String aggInstructionsInReducer, String otherInstructionsInReducer, 
			String[] outputVars, MatrixObjectNew[] outputMatrices, byte[] resultIndexes,
			int numReducers, int replication, String dimsUnknownFilePrefix) 
	throws Exception
	{
		String[] inputs = new String[inputMatrices.length];
		InputInfo[] inputInfos = new InputInfo[inputMatrices.length];
		long[] rlens = new long[inputMatrices.length];
		long[] clens = new long[inputMatrices.length];
		int[] brlens = new int[inputMatrices.length];
		int[] bclens = new int[inputMatrices.length];
		
		String[] outputs = new String[outputVars.length];
		OutputInfo[] outputInfos = new OutputInfo[outputVars.length];
		
		populateInputs(inputVars, inputMatrices, inputs, inputInfos, rlens, clens, brlens, bclens);
		populateOutputs(outputVars, outputMatrices, outputs, outputInfos);
		
		return runJob(inputs, inputInfos, rlens, clens, 
				brlens, bclens, recordReaderInstruction, instructionsInMapper, aggInstructionsInReducer, 
				otherInstructionsInReducer, numReducers, replication, resultIndexes, dimsUnknownFilePrefix,
				outputs, outputInfos);
	}
	
	public static JobReturn runJob(String[] inputs, InputInfo[] inputInfos, long[] rlens, long[] clens, 
			int[] brlens, int[] bclens, String recordReaderInstruction, String instructionsInMapper, String aggInstructionsInReducer, 
			String otherInstructionsInReducer, int numReducers, int replication, byte[] resultIndexes, String dimsUnknownFilePrefix,
			String[] outputs, OutputInfo[] outputInfos) 
	throws Exception
	{
		boolean inBlockRepresentation=MRJobConfiguration.deriveRepresentation(inputInfos);
		return runJob(inBlockRepresentation, inputs, inputInfos, rlens, clens, 
				brlens, bclens, recordReaderInstruction, instructionsInMapper, aggInstructionsInReducer, 
				otherInstructionsInReducer, numReducers, replication, resultIndexes, dimsUnknownFilePrefix,
				outputs, outputInfos); 
	}
	
	public static JobReturn runJob(boolean inBlockRepresentation, String[] inputs, InputInfo[] inputInfos, long[] rlens, long[] clens, 
			int[] brlens, int[] bclens, String instructionsInMapper, String aggInstructionsInReducer, 
			String otherInstructionsInReducer, int numReducers, int replication, byte[] resultIndexes, String dimsUnknownFilePrefix,
			String[] outputs, OutputInfo[] outputInfos) 
	throws Exception
	{
		return runJob(inBlockRepresentation, inputs, inputInfos, rlens, clens, 
				brlens, bclens, null, instructionsInMapper, aggInstructionsInReducer, 
				otherInstructionsInReducer, numReducers, replication, resultIndexes, dimsUnknownFilePrefix,
				outputs, outputInfos); 
	}
	
	@SuppressWarnings("unchecked")
	public static JobReturn runJob(boolean inBlockRepresentation, String[] inputs, InputInfo[] inputInfos, long[] rlens, long[] clens, 
			int[] brlens, int[] bclens, String recordReaderInstruction, String instructionsInMapper, String aggInstructionsInReducer, 
			String otherInstructionsInReducer, int numReducers, int replication, byte[] resultIndexes, String dimsUnknownFilePrefix, 
			String[] outputs, OutputInfo[] outputInfos) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(GMR.class);
		job.setJobName("G-MR");
		
		//whether use block representation or cell representation
		MRJobConfiguration.setMatrixValueClass(job, inBlockRepresentation);
	
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
		
		if(recordReaderInstruction!=null && !recordReaderInstruction.isEmpty())
		{
			assert(inputs.length<=2);
			PickByCountInstruction ins=(PickByCountInstruction) PickByCountInstruction.parseInstruction(recordReaderInstruction);
			PickFromCompactInputFormat.setKeyValueClasses(job, (Class<? extends WritableComparable>) inputInfos[ins.input1].inputKeyClass, 
					inputInfos[ins.input1].inputValueClass);
		    job.setInputFormat(PickFromCompactInputFormat.class);
		    PickFromCompactInputFormat.setZeroValues(job, (NumItemsByEachReducerMetaData)inputInfos[ins.input1].metadata);
		    
			if(ins.isValuePick)
			{
				double[] probs=MapReduceTool.readColumnVectorFromHDFS(inputs[ins.input2], inputInfos[ins.input2], rlens[ins.input2], 
						clens[ins.input2], brlens[ins.input2], bclens[ins.input2]);
			    PickFromCompactInputFormat.setPickRecordsInEachPartFile(job, (NumItemsByEachReducerMetaData) inputInfos[ins.input1].metadata, probs);
			    
			    realinputs=new String[inputs.length-1];
				realinputInfos=new InputInfo[inputs.length-1];
				realrlens=new long[inputs.length-1];
				realclens=new long[inputs.length-1];
				realbrlens=new int[inputs.length-1];
				realbclens=new int[inputs.length-1];
				realIndexes=new byte[inputs.length-1];
				byte realIndex=0;
				for(byte i=0; i<inputs.length; i++)
				{
					if(i==ins.input2)
						continue;
					realinputs[realIndex]=inputs[i];
					realinputInfos[realIndex]=inputInfos[i];
					if(i==ins.input1)
					{
						realrlens[realIndex]=rlens[ins.input2];
						realclens[realIndex]=clens[ins.input2];
						realbrlens[realIndex]=1;
						realbclens[realIndex]=1;
						realIndexes[realIndex]=ins.output;
					}else
					{	
						realrlens[realIndex]=rlens[i];
						realclens[realIndex]=clens[i];
						realbrlens[realIndex]=brlens[i];
						realbclens[realIndex]=bclens[i];
						realIndexes[realIndex]=i;
					}
					realIndex++;
				}
				
			}else
			{
			    PickFromCompactInputFormat.setPickRecordsInEachPartFile(job, (NumItemsByEachReducerMetaData) inputInfos[ins.input1].metadata, ins.cst, 1-ins.cst);
			    realrlens[ins.input1]=UtilFunctions.getLengthForInterQuantile((NumItemsByEachReducerMetaData)inputInfos[ins.input1].metadata, ins.cst);
				realclens[ins.input1]=clens[ins.input1];
				realbrlens[ins.input1]=1;
				realbclens[ins.input1]=1;
				realIndexes[ins.input1]=ins.output;
			}
		}
		
		//set up the input files and their format information
		MRJobConfiguration.setUpMultipleInputs(job, realIndexes, realinputs, realinputInfos, inBlockRepresentation, realbrlens, realbclens);
		
		//set up the dimensions of input matrices
		MRJobConfiguration.setMatricesDimensions(job, realIndexes, realrlens, realclens);
		MRJobConfiguration.setDimsUnknownFilePrefix(job, dimsUnknownFilePrefix);

		//set up the block size
		MRJobConfiguration.setBlocksSizes(job, realIndexes, realbrlens, realbclens);
		
		//set up unary instructions that will perform in the mapper
		MRJobConfiguration.setInstructionsInMapper(job, instructionsInMapper);
		
		//set up the aggregate instructions that will happen in the combiner and reducer
		MRJobConfiguration.setAggregateInstructions(job, aggInstructionsInReducer);
		
		//set up the instructions that will happen in the reducer, after the aggregation instrucions
		MRJobConfiguration.setInstructionsInReducer(job, otherInstructionsInReducer);
		
		//set up the number of reducers
		job.setNumReduceTasks(numReducers);
		
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);
		
		//set up what matrices are needed to pass from the mapper to reducer
		MRJobConfiguration.setUpOutputIndexesForMapper(job, realIndexes, instructionsInMapper, aggInstructionsInReducer, 
				otherInstructionsInReducer, resultIndexes);
		
		MatrixCharacteristics[] stats=MRJobConfiguration.computeMatrixCharacteristics(job, realIndexes, 
				instructionsInMapper, aggInstructionsInReducer, null, otherInstructionsInReducer, resultIndexes);
		
		// Update resultDimsUnknown based on computed "stats"
		byte[] dimsUnknown = new byte[resultIndexes.length];
		for ( int i=0; i < resultIndexes.length; i++ ) { 
			if ( stats[i].numRows == -1 || stats[i].numColumns == -1 ) {
				dimsUnknown[i] = (byte)1;
			}
			else {
				dimsUnknown[i] = (byte) 0;
			}
		}
		//MRJobConfiguration.updateResultDimsUnknown(job,resultDimsUnknown);
		
		//set up the multiple output files, and their format information
		MRJobConfiguration.setUpMultipleOutputs(job, resultIndexes, dimsUnknown, outputs, outputInfos, inBlockRepresentation, true);
		
		// configure mapper and the mapper output key value pairs
		job.setMapperClass(GMRMapper.class);
		if(numReducers==0)
		{
			job.setMapOutputKeyClass(Writable.class);
			job.setMapOutputValueClass(Writable.class);
		}else
		{
			job.setMapOutputKeyClass(MatrixIndexes.class);
			if(inBlockRepresentation)
				job.setMapOutputValueClass(TaggedMatrixBlock.class);
			else
				job.setMapOutputValueClass(TaggedMatrixPackedCell.class);
		}
		
		//set up combiner
		if(numReducers!=0 && aggInstructionsInReducer!=null 
				&& !aggInstructionsInReducer.isEmpty())
			job.setCombinerClass(GMRCombiner.class);
	
		//configure reducer
		job.setReducerClass(GMRReducer.class);
		//job.setReducerClass(PassThroughReducer.class);
		
		// By default, the job executes in "cluster" mode.
		// Determine if we can optimize and run it in "local" mode.
		MatrixCharacteristics[] inputStats = new MatrixCharacteristics[inputs.length];
		for ( int i=0; i < inputs.length; i++ ) {
			inputStats[i] = new MatrixCharacteristics(rlens[i], clens[i], brlens[i], bclens[i]);
		}
		ExecMode mode = RunMRJobs.getExecMode(JobType.GMR, inputStats); 
		if ( mode == ExecMode.LOCAL ) {
			job.set("mapred.job.tracker", "local");
			MRJobConfiguration.setStagingDir( job );
		}
		
		//System.out.println("Check mode = " + mode);
		
		//set unique working dir
		MRJobConfiguration.setUniqueWorkingDir(job, mode);
		
		
		RunningJob runjob=JobClient.runJob(job);
		
		Group group=runjob.getCounters().getGroup(MRJobConfiguration.NUM_NONZERO_CELLS);
		//MatrixCharacteristics[] stats=new MatrixCharacteristics[resultIndexes.length];
		for(int i=0; i<resultIndexes.length; i++) {
			// number of non-zeros
			stats[i].nonZero=group.getCounter(Integer.toString(i));
		}
		
		String dir = dimsUnknownFilePrefix + "/" + runjob.getID().toString() + "_dimsFile";
		stats = MapReduceTool.processDimsFiles(dir, stats);
		MapReduceTool.deleteFileIfExistOnHDFS(dir);

		/* Process different counters */
		
/*		Group group=runjob.getCounters().getGroup(MRJobConfiguration.NUM_NONZERO_CELLS);
		Group rowgroup, colgroup;
		
		for(int i=0; i<resultIndexes.length; i++)
		{
			// number of non-zeros
			stats[i].nonZero=group.getCounter(Integer.toString(i));
		//	System.out.println("result #"+resultIndexes[i]+" ===>\n"+stats[i]);
			
			// compute dimensions for output matrices whose dimensions are unknown at compilation time 
			if ( stats[i].numRows == -1 || stats[i].numColumns == -1 ) {
				if ( resultDimsUnknown[i] != (byte) 1 )
					throw new DMLRuntimeException("Unexpected error after executing GMR Job");
			
				rowgroup = runjob.getCounters().getGroup("max_rowdim_"+i);
				colgroup = runjob.getCounters().getGroup("max_coldim_"+i);
				int maxrow, maxcol;
				maxrow = maxcol = 0;
				for ( int rid=0; rid < numReducers; rid++ ) {
					if ( maxrow < (int) rowgroup.getCounter(Integer.toString(rid)) )
						maxrow = (int) rowgroup.getCounter(Integer.toString(rid));
					if ( maxcol < (int) colgroup.getCounter(Integer.toString(rid)) )
						maxcol = (int) colgroup.getCounter(Integer.toString(rid)) ;
				}
				//System.out.println("Resulting Rows = " + maxrow + ", Cols = " + maxcol );
				stats[i].numRows = maxrow;
				stats[i].numColumns = maxcol;
			}
		}
*/		
		
		return new JobReturn(stats, outputInfos, runjob.isSuccessful());
	}
}
