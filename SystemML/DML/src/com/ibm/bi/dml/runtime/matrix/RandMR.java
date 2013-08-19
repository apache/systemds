package com.ibm.bi.dml.runtime.matrix;

import java.io.PrintWriter;
import java.util.HashSet;
import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math.random.Well1024a;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.Counters.Group;

import com.ibm.bi.dml.hops.RandOp;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.lops.runtime.RunMRJobs;
import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RandInstruction;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.TaggedMatrixBlock;
import com.ibm.bi.dml.runtime.matrix.mapred.GMRCombiner;
import com.ibm.bi.dml.runtime.matrix.mapred.GMRReducer;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.matrix.mapred.RandMapper;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration.MatrixChar_N_ReducerGroups;
import com.ibm.bi.dml.runtime.util.MapReduceTool;


/**
 * <p>Rand MapReduce job which creates random objects.</p>
 * 
 */
public class RandMR
{
	/**
	 * <p>Starts a Rand MapReduce job which will produce one or more random objects.</p>
	 * 
	 * @param numRows number of rows for each random object
	 * @param numCols number of columns for each random object
	 * @param blockRowSize number of rows in a block for each random object
	 * @param blockColSize number of columns in a block for each random object
	 * @param minValue minimum of the random values for each random object
	 * @param maxValue maximum of the random values for each random object
	 * @param sparsity sparsity for each random object
	 * @param pdf probability density function for each random object
	 * @param replication file replication
	 * @param inputs input file for each random object
	 * @param outputs output file for each random object
	 * @param outputInfos output information for each random object
	 * @param instructionsInMapper instruction for each random object
	 * @param resultIndexes result indexes for each random object
	 * @return matrix characteristics for each random object
	 * @throws Exception if an error occurres in the MapReduce phase
	 */
	private static final Log LOG = LogFactory.getLog(RandMR.class.getName());
	
	public static JobReturn runJob(MRJobInstruction inst, String[] randInstructions, 
			String instructionsInMapper, String aggInstructionsInReducer, String otherInstructionsInReducer, 
			int numReducers, int replication, byte[] resultIndexes, String dimsUnknownFilePrefix, 
			String[] outputs, OutputInfo[] outputInfos) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(RandMR.class);
		job.setJobName("Rand-MR");
		
		//whether use block representation or cell representation
		MRJobConfiguration.setMatrixValueClass(job, true);
		
		
		byte[] realIndexes=new byte[randInstructions.length];
		for(byte b=0; b<realIndexes.length; b++)
			realIndexes[b]=b;
		
		String[] inputs=new String[randInstructions.length];
		InputInfo[] inputInfos = new InputInfo[randInstructions.length];
		long[] rlens=new long[randInstructions.length];
		long[] clens=new long[randInstructions.length];
		int[] brlens=new int[randInstructions.length];
		int[] bclens=new int[randInstructions.length];
		
		FileSystem fs = FileSystem.get(job);
		Random random=new Random();
		Well1024a bigrand=new Well1024a();
		String randInsStr="";
		int numblocks=0;
		int maxbrlen=-1, maxbclen=-1;
		
		for(int i = 0; i < randInstructions.length; i++)
		{
			randInsStr=randInsStr+Lops.INSTRUCTION_DELIMITOR+randInstructions[i];
			RandInstruction ins=(RandInstruction)RandInstruction.parseInstruction(randInstructions[i]);
			inputs[i]=ins.baseDir + System.currentTimeMillis()+".randinput";//+random.nextInt();
			FSDataOutputStream fsOut = fs.create(new Path(inputs[i]));
			PrintWriter pw = new PrintWriter(fsOut);
			
			//for obj reuse and preventing repeated buffer re-allocations
			StringBuilder sb = new StringBuilder();
			
			//seed generation
			long lSeed = ins.seed;
			if(lSeed == RandOp.UNSPECIFIED_SEED)
				lSeed = RandOp.generateRandomSeed();
			random.setSeed(lSeed);
			int[] seeds=new int[32];
			for(int s=0; s<seeds.length; s++)
				seeds[s]=random.nextInt();
			bigrand.setSeed(seeds);
			
			LOG.trace("Processing RandMR with seed = "+lSeed+".");

			rlens[i]=ins.rows;
			clens[i]=ins.cols;
			brlens[i] = ins.rowsInBlock;
			bclens[i] = ins.colsInBlock;
			maxbrlen = Math.max(maxbrlen, brlens[i]);
			maxbclen = Math.max(maxbclen, bclens[i]);
			
			for(long r = 0; r < ins.rows; r += brlens[i])
			{
				long curBlockRowSize = Math.min(brlens[i], (ins.rows - r));
				for(long c = 0; c < ins.cols; c += bclens[i])
				{
					long curBlockColSize = Math.min(bclens[i], (ins.cols - c));
					
					sb.append((r / brlens[i]) + 1);
					sb.append(',');
					sb.append((c / bclens[i]) + 1);
					sb.append(',');
					sb.append(curBlockRowSize);
					sb.append(',');
					sb.append(curBlockColSize);
					sb.append(',');
					sb.append(bigrand.nextLong());
					pw.println(sb.toString());
					sb.setLength(0);
					numblocks++;
				}
			}
			pw.close();
			fsOut.close();
			inputInfos[i] = InputInfo.TextCellInputInfo;
		}
		randInsStr=randInsStr.substring(1);//remove the first ","
		RunningJob runjob;
		MatrixCharacteristics[] stats;
		try{
			//set up the block size
			MRJobConfiguration.setBlocksSizes(job, realIndexes, brlens, bclens);
			
			//set up the input files and their format information
			MRJobConfiguration.setUpMultipleInputs(job, realIndexes, inputs, inputInfos, true, brlens, bclens, false);
			
			//set up the dimensions of input matrices
			MRJobConfiguration.setMatricesDimensions(job, realIndexes, rlens, clens);
			MRJobConfiguration.setDimsUnknownFilePrefix(job, dimsUnknownFilePrefix);
			
			//set up the block size
			MRJobConfiguration.setBlocksSizes(job, realIndexes, brlens, bclens);
			
			//set up the rand Instructions
			MRJobConfiguration.setRandInstructions(job, randInsStr);
			
			//set up unary instructions that will perform in the mapper
			MRJobConfiguration.setInstructionsInMapper(job, instructionsInMapper);
			
			//set up the aggregate instructions that will happen in the combiner and reducer
			MRJobConfiguration.setAggregateInstructions(job, aggInstructionsInReducer);
			
			//set up the instructions that will happen in the reducer, after the aggregation instrucions
			MRJobConfiguration.setInstructionsInReducer(job, otherInstructionsInReducer);
			
			
			//set up the replication factor for the results
			job.setInt("dfs.replication", replication);
			
			JobClient client=new JobClient(job);
			int capacity=client.getClusterStatus().getMaxMapTasks();
			int dfsblocksize=job.getInt("dfs.block.size", 67108864);
			int nmapers=Math.min((int)(8*maxbrlen*maxbclen*(long)numblocks/(long)dfsblocksize), capacity);
			job.setNumMapTasks(nmapers);
			
			//set up what matrices are needed to pass from the mapper to reducer
			HashSet<Byte> mapoutputIndexes=MRJobConfiguration.setUpOutputIndexesForMapper(job, realIndexes,  randInsStr, instructionsInMapper, null, aggInstructionsInReducer, otherInstructionsInReducer, resultIndexes);
			
			MatrixChar_N_ReducerGroups ret=MRJobConfiguration.computeMatrixCharacteristics(job, realIndexes, randInsStr,
					instructionsInMapper, null, aggInstructionsInReducer, null, otherInstructionsInReducer, 
					resultIndexes, mapoutputIndexes, false);
			stats=ret.stats;
			
			//set up the number of reducers
			MRJobConfiguration.setNumReducers(job, ret.numReducerGroups, numReducers);
			
			// print the complete MRJob instruction
			if (LOG.isTraceEnabled())
				inst.printCompelteMRJobInstruction(stats);
			
			// Update resultDimsUnknown based on computed "stats"
			byte[] resultDimsUnknown = new byte[resultIndexes.length]; 
			for ( int i=0; i < resultIndexes.length; i++ ) { 
				if ( stats[i].numRows == -1 || stats[i].numColumns == -1 ) {
					resultDimsUnknown[i] = (byte) 1;
				}
				else {
					resultDimsUnknown[i] = (byte) 0;
				}
			}
			
			//set up the multiple output files, and their format information
			MRJobConfiguration.setUpMultipleOutputs(job, resultIndexes, resultDimsUnknown, outputs, outputInfos, true, true);
			
			// configure mapper and the mapper output key value pairs
			job.setMapperClass(RandMapper.class);
			if(numReducers==0)
			{
				job.setMapOutputKeyClass(Writable.class);
				job.setMapOutputValueClass(Writable.class);
			}else
			{
				job.setMapOutputKeyClass(MatrixIndexes.class);
				job.setMapOutputValueClass(TaggedMatrixBlock.class);
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
			ExecMode mode = RunMRJobs.getExecMode(JobType.RAND, inputStats); 
			if ( mode == ExecMode.LOCAL ) {
				job.set("mapred.job.tracker", "local");
				MRJobConfiguration.setStagingDir( job );
			}

			//set unique working dir
			MRJobConfiguration.setUniqueWorkingDir(job, mode);
			
			
			runjob=JobClient.runJob(job);
			
			/* Process different counters */
			
			Group group=runjob.getCounters().getGroup(MRJobConfiguration.NUM_NONZERO_CELLS);
			for(int i=0; i<resultIndexes.length; i++) {
				// number of non-zeros
				stats[i].nonZero=group.getCounter(Integer.toString(i));
				//	System.out.println("result #"+resultIndexes[i]+" ===>\n"+stats[i]);
			}
			
			String dir = dimsUnknownFilePrefix + "/" + runjob.getID().toString() + "_dimsFile";
			stats = MapReduceTool.processDimsFiles(dir, stats);
			MapReduceTool.deleteFileIfExistOnHDFS(dir);
			
/*			Group rowgroup, colgroup;
			for(int i=0; i<resultIndexes.length; i++)
			{
				// number of non-zeros
				stats[i].nonZero=group.getCounter(Integer.toString(i));
			//	System.out.println("result #"+resultIndexes[i]+" ===>\n"+stats[i]);
				
				// compute dimensions for output matrices whose dimensions are unknown at compilation time 
				if ( stats[i].numRows == -1 || stats[i].numColumns == -1 ) {
					if ( resultDimsUnknown[i] != (byte) 1 )
						throw new DMLRuntimeException("Unexpected error after executing Rand Job");
				
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
		}finally
		{
			for(String input: inputs)
				MapReduceTool.deleteFileIfExistOnHDFS(new Path(input), job);
		}
		
		return new JobReturn(stats, outputInfos, runjob.isSuccessful());
	}
}
