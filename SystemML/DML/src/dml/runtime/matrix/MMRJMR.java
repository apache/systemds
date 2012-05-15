package dml.runtime.matrix;

import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.Counters.Group;

import dml.lops.compile.JobType;
import dml.lops.runtime.RunMRJobs;
import dml.lops.runtime.RunMRJobs.ExecMode;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;
import dml.runtime.matrix.io.TaggedMatrixBlock;
import dml.runtime.matrix.io.TaggedMatrixCell;
import dml.runtime.matrix.io.TripleIndexes;
import dml.runtime.matrix.mapred.MMRJMRMapper;
import dml.runtime.matrix.mapred.MMRJMRReducer;
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
 * otherInstructionsInReducer: the mixed operations that need to be performed on matrices after the aggregate operations
 * numReducers: the number of reducers
 * replication: the replication factor for the output
 * resulltIndexes: the indexes of the result matrices that needs to be outputted.
 * outputs: the names for the output directories, one for each result index
 * outputInfos: output format information for the output matrices
 */
public class MMRJMR {
	
	public static JobReturn runJob(boolean inBlockRepresentation, String[] inputs, InputInfo[] inputInfos, 
			long[] rlens, long[] clens, int[] brlens, int[] bclens, String instructionsInMapper, 
			String aggInstructionsInReducer, String aggBinInstrction, String otherInstructionsInReducer, 
			int numReducers, int replication, byte[] resultIndexes, byte[] resultDimsUnknown, 
			String[] outputs, OutputInfo[] outputInfos) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(MMRJMR.class);
		
		job.setJobName("MMRJ-MR");
		
		if(numReducers<=0)
			throw new Exception("MMRJ-MR has to have at least one reduce task!");
		
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
		
		//set up the instructions that will happen in the reducer, after the aggregation instrucions
		MRJobConfiguration.setInstructionsInReducer(job, otherInstructionsInReducer);
		
		//set up the number of reducers
		job.setNumReduceTasks(numReducers);
		
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);
		
		// byte[] resultIndexes=new byte[]{AggregateBinaryInstruction.parseMRInstruction(aggBinInstrction).output};
		
		//set up what matrices are needed to pass from the mapper to reducer
		MRJobConfiguration.setUpOutputIndexesForMapper(job, realIndexes,  instructionsInMapper, aggInstructionsInReducer, 
				aggBinInstrction, resultIndexes );
		
		//set up the multiple output files, and their format information
		MRJobConfiguration.setUpMultipleOutputs(job, resultIndexes, resultDimsUnknown, outputs, outputInfos, inBlockRepresentation);
		
		// configure mapper
		job.setMapperClass(MMRJMRMapper.class);
		job.setMapOutputKeyClass(TripleIndexes.class);
		if(inBlockRepresentation)
			job.setMapOutputValueClass(TaggedMatrixBlock.class);
		else
			job.setMapOutputValueClass(TaggedMatrixCell.class);
		job.setOutputKeyComparatorClass(TripleIndexes.Comparator.class);
		job.setPartitionerClass(TripleIndexes.FirstTwoIndexesPartitioner.class);
		
		//configure combiner
		//TODO: cannot set up combiner, because it will destroy the stable numerical algorithms 
		// for sum or for central moments 
		
	//	if(aggInstructionsInReducer!=null && !aggInstructionsInReducer.isEmpty())
	//		job.setCombinerClass(MMCJMRCombiner.class);
		
		MatrixCharacteristics[] stats=MRJobConfiguration.computeMatrixCharacteristics(job, realIndexes, 
				instructionsInMapper, aggInstructionsInReducer, aggBinInstrction, otherInstructionsInReducer, resultIndexes);
		
		//configure reducer
		job.setReducerClass(MMRJMRReducer.class);
		
		// By default, the job executes in "cluster" mode.
		// Determine if we can optimize and run it in "local" mode.
		MatrixCharacteristics[] inputStats = new MatrixCharacteristics[inputs.length];
		for ( int i=0; i < inputs.length; i++ ) {
			inputStats[i] = new MatrixCharacteristics(rlens[i], clens[i], brlens[i], bclens[i]);
		}
		ExecMode mode = RunMRJobs.getExecMode(JobType.MMRJ, inputStats); 
		if ( mode == ExecMode.LOCAL ) {
			job.set("mapred.job.tracker", "local");
		}

		//set unique working dir
		MRJobConfiguration.setUniqueWorkingDir(job, mode);
		
		
		RunningJob runjob=JobClient.runJob(job);
		
		/* Process different counters */
		
		Group group=runjob.getCounters().getGroup(MRJobConfiguration.NUM_NONZERO_CELLS);
		Group rowgroup, colgroup;
		
		for(int i=0; i<resultIndexes.length; i++)
		{
			// number of non-zeros
			stats[i].nonZeros=group.getCounter(Byte.toString(resultIndexes[i]));
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
		
		//Group group=runjob.getCounters().getGroup(MRJobConfiguration.NUM_NONZERO_CELLS);
		//for(int i=0; i<resultIndexes.length; i++)
		//{
		//	stats[i].nonZeros=group.getCounter(Byte.toString(resultIndexes[i]));
		////	System.out.println("result #"+resultIndexes[i]+" ===>\n"+stats[i]);
		//}
		
		return new JobReturn(stats, outputInfos, runjob.isSuccessful());
	}

}
