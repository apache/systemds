/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix;

import java.util.ArrayList;
import java.util.HashSet;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.Counters.Group;

import com.ibm.bi.dml.lops.AppendM;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.MapMult;
import com.ibm.bi.dml.lops.MapMult.CacheType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.lops.runtime.RunMRJobs;
import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
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
import com.ibm.bi.dml.runtime.matrix.mapred.MRBaseForCommonInstructions;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration.ConvertTarget;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration.MatrixChar_N_ReducerGroups;
import com.ibm.bi.dml.runtime.matrix.sort.PickFromCompactInputFormat;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

 
public class GMR
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
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
	private static final Log LOG = LogFactory.getLog(GMR.class.getName());
	
	@SuppressWarnings("unchecked")
	public static JobReturn runJob(MRJobInstruction inst, String[] inputs, InputInfo[] inputInfos, long[] rlens, long[] clens, 
			int[] brlens, int[] bclens, 
			boolean[] partitioned, PDataPartitionFormat[] pformats, int[] psizes,
			String recordReaderInstruction, String instructionsInMapper, String aggInstructionsInReducer, 
			String otherInstructionsInReducer, int numReducers, int replication, boolean jvmReuse, byte[] resultIndexes, String dimsUnknownFilePrefix, 
			String[] outputs, OutputInfo[] outputInfos) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(GMR.class);
		job.setJobName("G-MR");
		
		boolean inBlockRepresentation=MRJobConfiguration.deriveRepresentation(inputInfos);

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
		
		setupDistributedCache(job, instructionsInMapper, realinputs, realrlens, realclens);

		//set up the input files and their format information
		boolean[] distCacheOnly = getDistCacheOnlyInputs(realIndexes, recordReaderInstruction, instructionsInMapper, aggInstructionsInReducer, otherInstructionsInReducer);
		MRJobConfiguration.setUpMultipleInputs(job, realIndexes, realinputs, realinputInfos, realbrlens, realbclens, distCacheOnly, 
				                             true, inBlockRepresentation? ConvertTarget.BLOCK: ConvertTarget.CELL);
		MRJobConfiguration.setInputPartitioningInfo(job, pformats);
		
		//set up the dimensions of input matrices
		MRJobConfiguration.setMatricesDimensions(job, realIndexes, realrlens, realclens);
		MRJobConfiguration.setDimsUnknownFilePrefix(job, dimsUnknownFilePrefix);

		//set up the block size
		MRJobConfiguration.setBlocksSizes(job, realIndexes, realbrlens, realbclens);
		
		//set up unary instructions that will perform in the mapper
		MRJobConfiguration.setInstructionsInMapper(job, instructionsInMapper);
		
		//set up the aggregate instructions that will happen in the combiner and reducer
		MRJobConfiguration.setAggregateInstructions(job, aggInstructionsInReducer);
		
		//set up the instructions that will happen in the reducer, after the aggregation instructions
		MRJobConfiguration.setInstructionsInReducer(job, otherInstructionsInReducer);
		
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);

		//set up jvm reuse (incl. reuse of loaded dist cache matrices)
		if( jvmReuse )
			job.setNumTasksToExecutePerJvm(-1);
		
		//set up what matrices are needed to pass from the mapper to reducer
		HashSet<Byte> mapoutputIndexes=MRJobConfiguration.setUpOutputIndexesForMapper(job, realIndexes, instructionsInMapper, aggInstructionsInReducer, 
				otherInstructionsInReducer, resultIndexes);
		
		MatrixChar_N_ReducerGroups ret=MRJobConfiguration.computeMatrixCharacteristics(job, realIndexes, 
				instructionsInMapper, aggInstructionsInReducer, null, otherInstructionsInReducer, resultIndexes, mapoutputIndexes, false);
		
		MatrixCharacteristics[] stats=ret.stats;
		
		//set up the number of reducers
		MRJobConfiguration.setNumReducers(job, ret.numReducerGroups, numReducers);
		
		// Print the complete instruction
		if (LOG.isTraceEnabled())
			inst.printCompleteMRJobInstruction(stats);
		
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
		{
			job.setCombinerClass(GMRCombiner.class);
		}
	
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


	
	private static void setupDistributedCache(JobConf job, String instructionsInMapper, String[] inputs, long[] rlens, long[] clens) {
		if ( instructionsInMapper != null && instructionsInMapper != "" && InstructionUtils.isDistributedCacheUsed(instructionsInMapper) ) {
			String indexString = ""; // input indices to be placed in Distributed Cache (concatenated) 
			String pathString = "";  // input paths to be placed in Distributed Cache (concatenated) 
			ArrayList<String> pathList = new ArrayList<String>(); // list of paths to be placed in Distributed cache
			
			byte index;
			String[] inst = instructionsInMapper.split(Instruction.INSTRUCTION_DELIM);
			for(int i=0; i < inst.length; i++) {
				if ( inst[i].contains(MapMult.OPCODE) || inst[i].contains(AppendM.OPCODE) ) {
					// example: MR.mvmult.0.1.2
					
					// Determine the index that points to a vector
					String[] parts = inst[i].split(Instruction.OPERAND_DELIM);
					byte in1 = Byte.parseByte(parts[2].split(Instruction.DATATYPE_PREFIX)[0]);
					byte in2 = Byte.parseByte(parts[3].split(Instruction.DATATYPE_PREFIX)[0]);
					boolean rightCache = inst[i].contains(MapMult.OPCODE)?CacheType.valueOf(parts[5]).isRightCache():true; //4 is out
					if ( rightCache )
						index = in2; // input2 is in dist cache
					else 
						index = in1; // input1 is in dist cache
					
					if ( !pathList.contains(index) ) {
						pathList.add(inputs[index]);
						if ( indexString.equalsIgnoreCase("") ) {
							indexString += index;
							pathString += inputs[index];
						} 
						else {
							indexString += Instruction.INSTRUCTION_DELIM + index;
							pathString += Instruction.INSTRUCTION_DELIM + inputs[index];
						}
					}
				}
			}
			
			MRJobConfiguration.setupDistCacheInputs(job, indexString, pathString, pathList);
			
			//clean in-memory cache (prevent job interference in local mode)
			if( MRJobConfiguration.isLocalJobTracker(job) )
				MRBaseForCommonInstructions.resetDistCache();
		}
	}

	/**
	 * Determine which indices are only used as inputs through distribtued cache and hence would
	 * be redundant job inputs.
	 * 
	 * @param realIndexes
	 * @param inst1
	 * @param inst2
	 * @param inst3
	 * @param inst4
	 * @return
	 */
	private static boolean[] getDistCacheOnlyInputs(byte[] realIndexes, String inst1, String inst2, String inst3, String inst4)
	{
		boolean[] ret = new boolean[realIndexes.length];
		String[] inst = new String[]{inst1, inst2, inst3, inst4};
		
		//for all result indexes
		for( int i=0; i<ret.length; i++ ){
			byte index = realIndexes[i];
			String indexStr = index+Lop.DATATYPE_PREFIX+DataType.MATRIX.toString();
			
			boolean distCacheOnly = true;
			boolean use = false;
			for( String linst : inst ){ //for all instruction categories
				if(linst!=null){
					String[] alinst = linst.split(Lop.INSTRUCTION_DELIMITOR);
					for( String tmp : alinst ) //for each individual instruction
					{
						boolean lcache = false;
						if ( tmp.contains(MapMult.OPCODE) || tmp.contains(AppendM.OPCODE) ) {
							String[] parts = tmp.split(Instruction.OPERAND_DELIM);
							byte in1 = Byte.parseByte(parts[2].split(Instruction.DATATYPE_PREFIX)[0]);
							byte in2 = Byte.parseByte(parts[3].split(Instruction.DATATYPE_PREFIX)[0]);
							boolean rightCache = tmp.contains(MapMult.OPCODE)?CacheType.valueOf(parts[5]).isRightCache():true; //4 is out
							lcache = rightCache ? (index==in2 && index!=in1) : (index==in1&& index!=in2);
						}	
						distCacheOnly &= (lcache || !tmp.contains(indexStr));
						use |= tmp.contains(indexStr);
					}
				}
			}
			//probe for use in order to account for write only jobs
			ret[i] = distCacheOnly && use;
		}
		
		return ret;
	}
}
