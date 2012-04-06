package dml.runtime.matrix;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.Counters.Group;

import dml.lops.compile.JobType;
import dml.lops.runtime.RunMRJobs;
import dml.lops.runtime.RunMRJobs.ExecMode;
import dml.runtime.instructions.MRInstructionParser;
import dml.runtime.instructions.MRInstructions.AggregateBinaryInstruction;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixCell;
import dml.runtime.matrix.io.OutputInfo;
import dml.runtime.matrix.io.TaggedFirstSecondIndexes;
import dml.runtime.matrix.mapred.MMCJMRMapper;
import dml.runtime.matrix.mapred.MMCJMRReducerWithAggregator;
import dml.runtime.matrix.mapred.MRJobConfiguration;
import dml.utils.DMLRuntimeException;

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
public class MMCJMR {

	protected static final Log LOG = LogFactory.getLog(MMCJMR.class);
	
	@SuppressWarnings("deprecation")
	public static JobReturn runJob(boolean inBlockRepresentation, String[] inputs, InputInfo[] inputInfos, long[] rlens, long[] clens, 
			int[] brlens, int[] bclens, String instructionsInMapper, 
			String aggInstructionsInReducer, String aggBinInstrction, int numReducers, 
			int replication, byte resultDimsUnknown, String output, OutputInfo outputinfo) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(MMCJMR.class);
		
		byte []resultDimsUnknown_arr = new byte[1];
		resultDimsUnknown_arr[0] = resultDimsUnknown;
	//	MRJobConfiguration.updateResultDimsUnknown(job,resultDimsUnknown_arr);
		
		MatrixCharacteristics[] stats=commonSetup(job, inBlockRepresentation, inputs, inputInfos, rlens, clens, 
				brlens, bclens, instructionsInMapper, aggInstructionsInReducer, aggBinInstrction, numReducers, 
				replication, resultDimsUnknown, output, outputinfo);
		
		// Update resultDimsUnknown based on computed "stats"
		// There is always a single output
		if ( stats[0].numRows == -1 || stats[0].numColumns == -1 ) {
			if ( resultDimsUnknown != (byte) 1 ) {
				throw new Exception("Unexpected error while configuring GMR job.");
			}
		}
		else {
			resultDimsUnknown = (byte) 0;
		}
		
		//get the total amount of jvm memory
		int partialAggCacheSize=MRJobConfiguration.getJVMMaxMemSize(job);
		
	//	LOG.info("total mem size: "+partialAggCacheSize);
		
		//take away the misc memory requirement by hadoop
		partialAggCacheSize-=MRJobConfiguration.getMiscMemRequired(job);
		
		//take away some mall memory needed for run the reduce function
		partialAggCacheSize-=300000000;//200MB
		
	//	LOG.info("after misc use for hadoop: "+partialAggCacheSize);
		
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
		
		LOG.info("dim1: "+dim1);
		LOG.info("dim2: "+dim2);
		
		long blockSize1=77+8*dim1.numRowsPerBlock*dim1.numColumnsPerBlock;
		long blockSize2=77+8*dim2.numRowsPerBlock*dim2.numColumnsPerBlock;
		long blockSizeResult=77+8*dim1.numRowsPerBlock*dim2.numColumnsPerBlock;
		
	//	LOG.info("block sizes: "+blockSize1+", "+blockSize2+","+blockSizeResult);
		
		long cacheSize;
		//cache the first result
		if(dim1.numRows<dim2.numColumns)
		{
			long numBlocks=(long)Math.ceil((double)dim1.numRows/(double)dim1.numRowsPerBlock);
	//		LOG.info("numBlocks: "+numBlocks);
			cacheSize=numBlocks*(20+blockSize1)+32;
	//		LOG.info(numBlocks+"*(20+"+blockSize1+")+32");
		}
		else //cache the second result
		{
			long numBlocks=(long)Math.ceil((double)dim2.numColumns/(double) dim2.numColumnsPerBlock);
	//		LOG.info("numBlocks: "+numBlocks);
			cacheSize=numBlocks*(20+blockSize2)+32;
	//		LOG.info(numBlocks+")*(20+"+blockSize2+")+32");
		}
			
	//	LOG.info("cacheSize: "+cacheSize);
		
		partialAggCacheSize-=cacheSize;
		
		//the cached key-value pair
		partialAggCacheSize-=Math.max(blockSize1, blockSize2);
		
		//the cached single result
		partialAggCacheSize-=blockSizeResult;
		
		if(partialAggCacheSize<0)
			partialAggCacheSize=0;
		
		MRJobConfiguration.setPartialAggCacheSize(job, partialAggCacheSize);
		
		LOG.info("aggregator buffer size: "+partialAggCacheSize);
		
		RunningJob runjob=JobClient.runJob(job);
		
		/* Process different counters */
		
		// NOTE: MMCJ job always has only a single output. 
		// Hence, no need to scan resultIndexes[] like other jobs
		
		int outputIndex = 0;
		Byte outputMatrixID = MRInstructionParser.parseSingleInstruction(aggBinInstrction).output;
		
		Group group=runjob.getCounters().getGroup(MRJobConfiguration.NUM_NONZERO_CELLS);
		Group rowgroup, colgroup;
		
		// number of non-zeros
		stats[outputIndex].nonZeros=group.getCounter(Byte.toString(outputMatrixID));
		
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
		
		return new JobReturn(stats[outputIndex], outputinfo, runjob.isSuccessful());
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
		MRJobConfiguration.setUpMultipleInputs(job, realIndexes, inputs, inputInfos, inBlockRepresentation, brlens, bclens);
		
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
		
		//set up the number of reducers
		job.setNumReduceTasks(numReducers);
		
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);
		
		byte[] resultIndexes=new byte[]{MRInstructionParser.parseSingleInstruction(aggBinInstrction).output};
		byte[] resultDimsUnknown_Array = new byte[]{resultDimsUnknown};
		// byte[] resultIndexes=new byte[]{AggregateBinaryInstruction.parseMRInstruction(aggBinInstrction).output};
		
		//set up what matrices are needed to pass from the mapper to reducer
		MRJobConfiguration.setUpOutputIndexesForMapper(job, realIndexes,  instructionsInMapper, aggInstructionsInReducer, 
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
		
		MatrixCharacteristics[] stats=MRJobConfiguration.computeMatrixCharacteristics(job, realIndexes, 
				instructionsInMapper, aggInstructionsInReducer, aggBinInstrction, null, resultIndexes);
		
		//configure reducer
		job.setReducerClass(MMCJMRReducerWithAggregator.class);
		
		return stats;
	}
	
	@SuppressWarnings("deprecation")
	public static JobReturn runJob(boolean inBlockRepresentation, String[] inputs, InputInfo[] inputInfos, long[] rlens, long[] clens, 
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
		}
		
		RunningJob runjob=JobClient.runJob(job);
		
		/*
		 * Process different counters
		 *   NOTE: MMCJ job always has only a single output. 
		 *   Hence, no need to scan resultIndexes[] like other jobs
 		 */
		
		//Group group=runjob.getCounters().getGroup(MRJobConfiguration.NUM_NONZERO_CELLS);
		//stats[0].nonZeros=group.getCounter(Byte.toString(MRInstructionParser.parseSingleInstruction(aggBinInstrction).output));
		
		int outputIndex = 0;
		Byte outputMatrixID = MRInstructionParser.parseSingleInstruction(aggBinInstrction).output;
		
		Group group=runjob.getCounters().getGroup(MRJobConfiguration.NUM_NONZERO_CELLS);
		Group rowgroup, colgroup;
		
		// number of non-zeros
		stats[outputIndex].nonZeros=group.getCounter(Byte.toString(outputMatrixID));
		
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

		
		return new JobReturn(stats[outputIndex], outputinfo, runjob.isSuccessful());
	}
}
