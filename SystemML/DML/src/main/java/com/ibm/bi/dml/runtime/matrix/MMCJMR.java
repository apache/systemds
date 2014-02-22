/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix;

import java.util.HashSet;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.Counters.Group;

import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.runtime.instructions.MRInstructionParser;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateBinaryInstruction;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.MMCJMRMapper;
import com.ibm.bi.dml.runtime.matrix.mapred.MMCJMRReducerWithAggregator;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration.ConvertTarget;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration.MatrixChar_N_ReducerGroups;


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
 * aggBinInstrction: the aggregate binary instruction for the MMCJ operation
 * numReducers: the number of reducers
 * replication: the replication factor for the output
 * output: the path for the output file
 * outputInfo: information about output format
 */
public class MMCJMR 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private static final Log LOG = LogFactory.getLog(MMCJMR.class);
	
	public static JobReturn runJob(MRJobInstruction inst, String[] inputs, InputInfo[] inputInfos, long[] rlens, long[] clens, 
			int[] brlens, int[] bclens, String instructionsInMapper, 
			String aggInstructionsInReducer, String aggBinInstrction, int numReducers, 
			int replication, String output, OutputInfo outputinfo) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(MMCJMR.class);
		
		// TODO: check w/ yuanyuan. This job always runs in blocked mode, and hence derivation is not necessary.
		boolean inBlockRepresentation=MRJobConfiguration.deriveRepresentation(inputInfos);
		
		// by default, assume that dimensions of MMCJ's output are known at compile time
		byte resultDimsUnknown = (byte) 0;   
		MatrixCharacteristics[] stats=commonSetup(job, inBlockRepresentation, inputs, inputInfos, rlens, clens, 
				brlens, bclens, instructionsInMapper, aggInstructionsInReducer, aggBinInstrction, numReducers, 
				replication, resultDimsUnknown, output, outputinfo);
		
		// Print the complete instruction
		if (LOG.isTraceEnabled())
			inst.printCompelteMRJobInstruction(stats);
		
		// Update resultDimsUnknown based on computed "stats"
		// There is always a single output
		if ( stats[0].numRows == -1 || stats[0].numColumns == -1 ) {
			resultDimsUnknown = (byte) 1;
			
			// if the dimensions are unknown, then setup done in commonSetup() must be updated
			byte[] resultIndexes=new byte[]{MRInstructionParser.parseSingleInstruction(aggBinInstrction).output};
			byte[] resultDimsUnknown_Array = new byte[]{resultDimsUnknown};
			//set up the multiple output files, and their format information
			MRJobConfiguration.setUpMultipleOutputs(job, resultIndexes, resultDimsUnknown_Array, new String[]{output}, new OutputInfo[]{outputinfo}, inBlockRepresentation);
		}

		AggregateBinaryInstruction ins=(AggregateBinaryInstruction) MRInstructionParser.parseSingleInstruction(aggBinInstrction);
		MatrixCharacteristics dim1 = MRJobConfiguration.getMatrixCharactristicsForBinAgg(job, ins.input1);
		MatrixCharacteristics dim2 = MRJobConfiguration.getMatrixCharactristicsForBinAgg(job, ins.input2);
		
		if(dim1.numRowsPerBlock>dim1.numRows)
			dim1.numRowsPerBlock=(int) dim1.numRows;
		if(dim1.numColumnsPerBlock>dim1.numColumns)
			dim1.numColumnsPerBlock=(int) dim1.numColumns;
		if(dim2.numRowsPerBlock>dim2.numRows)
			dim2.numRowsPerBlock=(int) dim2.numRows;
		if(dim2.numColumnsPerBlock>dim2.numColumns)
			dim2.numColumnsPerBlock=(int) dim2.numColumns;
	
		long blockSize1=77+8*dim1.numRowsPerBlock*dim1.numColumnsPerBlock;
		long blockSize2=77+8*dim2.numRowsPerBlock*dim2.numColumnsPerBlock;
		long blockSizeResult=77+8*dim1.numRowsPerBlock*dim2.numColumnsPerBlock;
		
		long cacheSize = -1;
		//cache the first result
		if(dim1.numRows<dim2.numColumns)
		{
			long numBlocks=(long)Math.ceil((double)dim1.numRows/(double)dim1.numRowsPerBlock);
			cacheSize=numBlocks*(20+blockSize1)+32;
		}
		else //cache the second result
		{
			long numBlocks=(long)Math.ceil((double)dim2.numColumns/(double) dim2.numColumnsPerBlock);
			cacheSize=numBlocks*(20+blockSize2)+32;
		}
		//add known memory consumption (will be substracted from output buffer)
		cacheSize += 2* Math.max(blockSize1, blockSize2) //the cached key-value pair  (plus input instance)
				  +  blockSizeResult //the cached single result
				  +  MRJobConfiguration.getMiscMemRequired(job); //misc memory requirement by hadoop
		MRJobConfiguration.setMMCJCacheSize(job, (int)cacheSize);
		
		//set unique working dir
		MRJobConfiguration.setUniqueWorkingDir(job, ExecMode.CLUSTER); 
		
		//run mmcj job
		RunningJob runjob=JobClient.runJob(job);
		
		/* Process different counters */
		
		// NOTE: MMCJ job always has only a single output. 
		// Hence, no need to scan resultIndexes[] like other jobs
		
		int outputIndex = 0;
		Byte outputMatrixID = MRInstructionParser.parseSingleInstruction(aggBinInstrction).output;
		
		Group group=runjob.getCounters().getGroup(MRJobConfiguration.NUM_NONZERO_CELLS);
		
		// number of non-zeros
		stats[outputIndex].nonZero=group.getCounter(Byte.toString(outputMatrixID));

/*		Group rowgroup, colgroup;
		// compute dimensions for output matrices whose dimensions are unknown at compilation time 
		if ( stats[outputIndex].numRows == -1 || stats[outputIndex].numColumns == -1 ) {
			if ( resultDimsUnknown != (byte) 1 )
				throw new DMLRuntimeException("Unexpected error after executing MMCJ Job");
		
			rowgroup = runjob.getCounters().getGroup("max_rowdim_" + outputMatrixID );
			colgroup = runjob.getCounters().getGroup("max_coldim_" + outputMatrixID );
			int maxrow, maxcol;
			maxrow = maxcol = 0;
			for ( int rid=0; rid < numReducers; rid++ ) {
				if ( maxrow < (int) rowgroup.getCounter(Integer.toString(rid)) )
					maxrow = (int) rowgroup.getCounter(Integer.toString(rid));
				if ( maxcol < (int) colgroup.getCounter(Integer.toString(rid)) )
					maxcol = (int) colgroup.getCounter(Integer.toString(rid)) ;
			}
			//System.out.println("Resulting Rows = " + maxrow + ", Cols = " + maxcol );
			stats[outputIndex].numRows = maxrow;
			stats[outputIndex].numColumns = maxcol;
		}
*/		return new JobReturn(stats[outputIndex], outputinfo, runjob.isSuccessful());
	}
	
	private static MatrixCharacteristics[] commonSetup(JobConf job, boolean inBlockRepresentation, String[] inputs, InputInfo[] inputInfos, long[] rlens, long[] clens, 
			int[] brlens, int[] bclens, String instructionsInMapper, 
			String aggInstructionsInReducer, String aggBinInstrction, int numReducers, 
			int replication, byte resultDimsUnknown, String output, OutputInfo outputinfo)
	throws Exception
	{
		job.setJobName("MMCJ-MR");
		
		if(numReducers<=0)
			throw new Exception("MMCJ-MR has to have at least one reduce task!");
		
		//whether use block representation or cell representation
		MRJobConfiguration.setMatrixValueClass(job, inBlockRepresentation);
		
		byte[] realIndexes=new byte[inputs.length];
		for(byte b=0; b<realIndexes.length; b++)
			realIndexes[b]=b;
		
		//set up the input files and their format information
		MRJobConfiguration.setUpMultipleInputs(job, realIndexes, inputs, inputInfos, brlens, bclens, true, 
				inBlockRepresentation? ConvertTarget.BLOCK: ConvertTarget.CELL);
		
		//set up the dimensions of input matrices
		MRJobConfiguration.setMatricesDimensions(job, realIndexes, rlens, clens);
		
		//set up the block size
		MRJobConfiguration.setBlocksSizes(job, realIndexes, brlens, bclens);
		
		//set up unary instructions that will perform in the mapper
		MRJobConfiguration.setInstructionsInMapper(job, instructionsInMapper);
		
		//set up the aggregate instructions that will happen in the combiner and reducer
		MRJobConfiguration.setAggregateInstructions(job, aggInstructionsInReducer);
		
		//set up the aggregate binary operation for the mmcj job
		MRJobConfiguration.setAggregateBinaryInstructions(job, aggBinInstrction);
		
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);
		//job.setInt("DMLBlockSize", DMLTranslator.DMLBlockSize);  TODO MP

		byte[] resultIndexes=new byte[]{MRInstructionParser.parseSingleInstruction(aggBinInstrction).output};
		byte[] resultDimsUnknown_Array = new byte[]{resultDimsUnknown};
		// byte[] resultIndexes=new byte[]{AggregateBinaryInstruction.parseMRInstruction(aggBinInstrction).output};
		
		//set up what matrices are needed to pass from the mapper to reducer
		HashSet<Byte> mapoutputIndexes=MRJobConfiguration.setUpOutputIndexesForMapper(job, realIndexes,  instructionsInMapper, aggInstructionsInReducer, 
				aggBinInstrction, resultIndexes );
		
		//set up the multiple output files, and their format information
		MRJobConfiguration.setUpMultipleOutputs(job, resultIndexes, resultDimsUnknown_Array, new String[]{output}, new OutputInfo[]{outputinfo}, inBlockRepresentation);
		
		// configure mapper
		job.setMapperClass(MMCJMRMapper.class);
		job.setMapOutputKeyClass(TaggedFirstSecondIndexes.class);
		if(inBlockRepresentation)
			job.setMapOutputValueClass(MatrixBlock.class);
		else
			job.setMapOutputValueClass(MatrixCell.class);
		job.setOutputKeyComparatorClass(TaggedFirstSecondIndexes.Comparator.class);
		job.setPartitionerClass(TaggedFirstSecondIndexes.FirstIndexPartitioner.class);
		
		//configure combiner
		//TODO: cannot set up combiner, because it will destroy the stable numerical algorithms 
		// for sum or for central moments 
		
	//	if(aggInstructionsInReducer!=null && !aggInstructionsInReducer.isEmpty())
	//		job.setCombinerClass(MMCJMRCombiner.class);
		
		MatrixChar_N_ReducerGroups ret=MRJobConfiguration.computeMatrixCharacteristics(job, realIndexes, 
				instructionsInMapper, aggInstructionsInReducer, aggBinInstrction, null, resultIndexes, 
				mapoutputIndexes, true);
		
		//set up the number of reducers
		MRJobConfiguration.setNumReducers(job, ret.numReducerGroups, numReducers);
		
		//configure reducer
		// note: the alternative MMCJMRReducer is not maintained
		job.setReducerClass(MMCJMRReducerWithAggregator.class);
		
		
		return ret.stats;
	}
	
	/*public static JobReturn runJob(boolean inBlockRepresentation, String[] inputs, InputInfo[] inputInfos, long[] rlens, long[] clens, 
			int[] brlens, int[] bclens, String instructionsInMapper, 
			String aggInstructionsInReducer, String aggBinInstrction, int numReducers, 
			int replication, byte resultDimsUnknown, String output, OutputInfo outputinfo, int partialAggCacheSize) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(MMCJMR.class);
		MatrixCharacteristics[] stats=commonSetup(job, inBlockRepresentation, inputs, inputInfos, rlens, clens, 
				brlens, bclens, instructionsInMapper, aggInstructionsInReducer, aggBinInstrction, numReducers, 
				replication, resultDimsUnknown, output, outputinfo);
		
		MRJobConfiguration.setPartialAggCacheSize(job, partialAggCacheSize);
		
		// By default, the job executes in "cluster" mode.
		// Determine if we can optimize and run it in "local" mode.
		MatrixCharacteristics[] inputStats = new MatrixCharacteristics[inputs.length];
		for ( int i=0; i < inputs.length; i++ ) {
			inputStats[i] = new MatrixCharacteristics(rlens[i], clens[i], brlens[i], bclens[i]);
		}
		ExecMode mode = RunMRJobs.getExecMode(JobType.MMCJ, inputStats); 
		if ( mode == ExecMode.LOCAL ) {
			job.set("mapred.job.tracker", "local");
			MRJobConfiguration.setStagingDir( job );
		}
		
		//set unique working dir
		MRJobConfiguration.setUniqueWorkingDir(job, mode);
		
		
		RunningJob runjob=JobClient.runJob(job);
		
		
		 * Process different counters
		 *   NOTE: MMCJ job always has only a single output. 
		 *   Hence, no need to scan resultIndexes[] like other jobs
 		 
		
		//Group group=runjob.getCounters().getGroup(MRJobConfiguration.NUM_NONZERO_CELLS);
		//stats[0].nonZeros=group.getCounter(Byte.toString(MRInstructionParser.parseSingleInstruction(aggBinInstrction).output));
		
		int outputIndex = 0;
		Byte outputMatrixID = MRInstructionParser.parseSingleInstruction(aggBinInstrction).output;
		
		Group group=runjob.getCounters().getGroup(MRJobConfiguration.NUM_NONZERO_CELLS);
		
		// number of non-zeros
		stats[outputIndex].nonZero=group.getCounter(Byte.toString(outputMatrixID));
		
		return new JobReturn(stats[outputIndex], outputinfo, runjob.isSuccessful());
	}*/
}
