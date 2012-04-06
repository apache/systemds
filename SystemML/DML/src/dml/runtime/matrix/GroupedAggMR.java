package dml.runtime.matrix;

import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.Counters.Group;

import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;
import dml.runtime.matrix.io.TaggedInt;
import dml.runtime.matrix.io.WeightedCell;
import dml.runtime.matrix.mapred.GroupedAggMRMapper;
import dml.runtime.matrix.mapred.GroupedAggMRReducer;
import dml.runtime.matrix.mapred.MRJobConfiguration;

public class GroupedAggMR {

	public static JobReturn runJob(String[] inputs, InputInfo[] inputInfos, long[] rlens, long[] clens, 
			int[] brlens, int[] bclens, String grpAggInstructions, String simpleReduceInstructions/*only scalar or reorg instructions allowed*/, 
			int numReducers, int replication, byte[] resultIndexes,	String[] outputs, OutputInfo[] outputInfos) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(GroupedAggMR.class);
		job.setJobName("GroupedAgg-MR");
		
		//whether use block representation or cell representation
		//MRJobConfiguration.setMatrixValueClassForCM_N_COM(job, true);
		MRJobConfiguration.setMatrixValueClass(job, false);
	
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
		
		//set up the grouped aggregate instructions that will happen in the combiner and reducer
		MRJobConfiguration.setGroupedAggInstructions(job, grpAggInstructions);
		
		//set up the instructions that will happen in the reducer, after the aggregation instrucions
		MRJobConfiguration.setInstructionsInReducer(job, simpleReduceInstructions);
		
		//set up the number of reducers
		job.setNumReduceTasks(numReducers);
		
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);
		
		//set up what matrices are needed to pass from the mapper to reducer
		MRJobConfiguration.setUpOutputIndexesForMapper(job, realIndexes, null, null, 
				grpAggInstructions, resultIndexes);
		
		byte[] resultDimsUnknown=new byte[resultIndexes.length];
		// Update resultDimsUnknown based on computed "stats"
		for ( int i=0; i < resultIndexes.length; i++ )  
			resultDimsUnknown[i] = (byte) 2;
	//	MRJobConfiguration.updateResultDimsUnknown(job,resultDimsUnknown);
		
		//set up the multiple output files, and their format information
		MRJobConfiguration.setUpMultipleOutputs(job, resultIndexes, resultDimsUnknown, outputs, outputInfos, false);
		
		// configure mapper and the mapper output key value pairs
		job.setMapperClass(GroupedAggMRMapper.class);
		
		job.setMapOutputKeyClass(TaggedInt.class);
		job.setMapOutputValueClass(WeightedCell.class);
		
		//configure reducer
		job.setReducerClass(GroupedAggMRReducer.class);
		//job.setReducerClass(PassThroughReducer.class);
		
	//	MatrixCharacteristics[] stats=MRJobConfiguration.computeMatrixCharacteristics(job, realIndexes, 
	//			null, null, grpAggInstructions, null, resultIndexes);
		
		//TODO: need to recompute the statistics
		
		// By default, the job executes in "cluster" mode.
		// Determine if we can optimize and run it in "local" mode.
		//TODO: need to ask Shirish on how to set it
	/*	ExecMode mode = RunMRJobs.getExecMode(LopProperties.CM_COV, stats); 
		if ( mode == ExecMode.LOCAL ) {
			job.set("mapred.job.tracker", "local");
		}*/
		
		RunningJob runjob=JobClient.runJob(job);
		
		Group group=runjob.getCounters().getGroup(MRJobConfiguration.NUM_NONZERO_CELLS);
		Group rowgroup, colgroup;
		
		MatrixCharacteristics[] stats=new MatrixCharacteristics[resultIndexes.length];
		for(int i=0; i<resultIndexes.length; i++)
		{
			// number of non-zeros
			stats[i]=new MatrixCharacteristics();
			stats[i].nonZeros=group.getCounter(Integer.toString(i));
		//	System.out.println("result #"+resultIndexes[i]+" ===>\n"+stats[i]);
			
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
			//System.out.println("stats: "+stats[i]);
		}
		
		return new JobReturn(stats, outputInfos, runjob.isSuccessful());
	}
}
