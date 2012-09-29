package com.ibm.bi.dml.runtime.matrix;

import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.Counters.Group;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.TaggedInt;
import com.ibm.bi.dml.runtime.matrix.io.WeightedCell;
import com.ibm.bi.dml.runtime.matrix.mapred.GroupedAggMRMapper;
import com.ibm.bi.dml.runtime.matrix.mapred.GroupedAggMRReducer;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.MapReduceTool;


public class GroupedAggMR {

	public static JobReturn runJob(MRJobInstruction inst, String[] inputs, InputInfo[] inputInfos, long[] rlens, long[] clens, 
			int[] brlens, int[] bclens, String grpAggInstructions, String simpleReduceInstructions/*only scalar or reorg instructions allowed*/, 
			int numReducers, int replication, byte[] resultIndexes,	String dimsUnknownFilePrefix, String[] outputs, OutputInfo[] outputInfos) 
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
		MRJobConfiguration.setDimsUnknownFilePrefix(job, dimsUnknownFilePrefix);
		//set up the block size
		MRJobConfiguration.setBlocksSizes(job, realIndexes, realbrlens, realbclens);
		
		//set up the grouped aggregate instructions that will happen in the combiner and reducer
		MRJobConfiguration.setGroupedAggInstructions(job, grpAggInstructions);
		
		//set up the instructions that will happen in the reducer, after the aggregation instrucions
		MRJobConfiguration.setInstructionsInReducer(job, simpleReduceInstructions);
		
		//set up the number of reducers
		MRJobConfiguration.setNumReducers(job, numReducers, numReducers);
		
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);
		
		//set up what matrices are needed to pass from the mapper to reducer
		MRJobConfiguration.setUpOutputIndexesForMapper(job, realIndexes, null, null, 
				grpAggInstructions, resultIndexes);
		
		MatrixCharacteristics[] stats=new MatrixCharacteristics[resultIndexes.length];
		for( int i=0; i < resultIndexes.length; i++ )
			stats[i] = new MatrixCharacteristics();
		
		// Print the complete instruction
		if ( DMLScript.DEBUG )
			inst.printCompelteMRJobInstruction(stats);

		byte[] resultDimsUnknown=new byte[resultIndexes.length];
		// Update resultDimsUnknown based on computed "stats"
		for ( int i=0; i < resultIndexes.length; i++ )  
			resultDimsUnknown[i] = (byte) 2;
		
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
			MRJobConfiguration.setStagingDir( job );
		}*/
		
		
		
		ExecMode mode = ExecMode.CLUSTER; //default
		//set unique working dir
		MRJobConfiguration.setUniqueWorkingDir(job, mode); //TODO see above
		
		
		RunningJob runjob=JobClient.runJob(job);
		
		Group group=runjob.getCounters().getGroup(MRJobConfiguration.NUM_NONZERO_CELLS);
		for(int i=0; i<resultIndexes.length; i++) {
			// number of non-zeros
			stats[i]=new MatrixCharacteristics();
			stats[i].nonZero=group.getCounter(Integer.toString(i));
		}
		
		String dir = dimsUnknownFilePrefix + "/" + runjob.getID().toString() + "_dimsFile";
		stats = MapReduceTool.processDimsFiles(dir, stats);
		MapReduceTool.deleteFileIfExistOnHDFS(dir);
		
/*		Counters counters = runjob.getCounters();
		System.out.println("Counters size = " + counters.size());
		System.out.println("All Counters = " +counters.toString());
		Group maxrowGroup = counters.getGroup(MRJobConfiguration.MAX_ROW_DIMENSION);
		Group maxcolGroup = counters.getGroup(MRJobConfiguration.MAX_COL_DIMENSION);
		
		for (int i=0; i < resultIndexes.length; i++) {
			long r = maxrowGroup.getCounter(Integer.toString(i));
			long c = maxcolGroup.getCounter(Integer.toString(i));
			stats[i].numRows = (stats[i].numRows > r ? stats[i].numRows : r);
			stats[i].numColumns = (stats[i].numColumns > c ? stats[i].numColumns : c);
		}
		
		Group rowgroup, colgroup;
		
		for(int i=0; i<resultIndexes.length; i++)
		{
			// number of non-zeros
			stats[i]=new MatrixCharacteristics();
			stats[i].nonZero=group.getCounter(Integer.toString(i));
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
*/		
		return new JobReturn(stats, outputInfos, runjob.isSuccessful());
	}
	
}
