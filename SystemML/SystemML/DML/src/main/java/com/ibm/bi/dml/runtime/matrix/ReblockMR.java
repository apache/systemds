/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
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

import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.lops.runtime.RunMRJobs;
import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.data.TaggedAdaptivePartialBlock;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.matrix.mapred.ReblockMapper;
import com.ibm.bi.dml.runtime.matrix.mapred.ReblockReducer;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration.MatrixChar_N_ReducerGroups;
import com.ibm.bi.dml.yarn.ropt.YarnClusterAnalyzer;


/*
 * inputs: input matrices, the inputs are indexed by 0, 1, 2, .. based on the position in this string
 * inputInfos: the input format information for the input matrices
 * rlen: the number of rows for each matrix
 * clen: the number of columns for each matrix
 * brlen: the number of rows per block for each input matrix
 * bclen: the number of columns per block for each input matrix
 * instructionsInMapper: in Mapper, the set of unary operations that need to be performed on each input matrix
 * reblockInstructions: the reblock instructions 
 * otherInstructionsInReducer: the mixed operations that need to be performed on matrices
 * numReducers: the number of reducers
 * replication: the replication factor for the output
 * resulltIndexes: the indexes of the result matrices that needs to be outputted.
 * outputs: the names for the output directories, one for each result index
 * outputInfos: output format information for the output matrices
 */

public class ReblockMR 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private static final Log LOG = LogFactory.getLog(ReblockMR.class.getName());
	
	private ReblockMR() {
		//prevent instantiation via private constructor
	}
	
	public static JobReturn runJob(MRJobInstruction inst, String[] inputs, InputInfo[] inputInfos, long[] rlens, long[] clens, 
			int[] brlens, int[] bclens, long[] nnz, String instructionsInMapper, String reblockInstructions, 
			String otherInstructionsInReducer, int numReducers, int replication, boolean jvmReuse, byte[] resultIndexes, 
			String[] outputs, OutputInfo[] outputInfos) 
	throws Exception
	{
		JobConf job = new JobConf(ReblockMR.class);
		job.setJobName("Reblock-MR");
		
		byte[] realIndexes=new byte[inputs.length];
		for(byte b=0; b<realIndexes.length; b++)
			realIndexes[b]=b;
		
		//set up the input files and their format information
		//(internally used input converters: text2bc for text, identity for binary inputs)
		MRJobConfiguration.setUpMultipleInputsReblock(job, realIndexes, inputs, inputInfos, brlens, bclens);
		
		//set up the dimensions of input matrices
		MRJobConfiguration.setMatricesDimensions(job, realIndexes, rlens, clens, nnz);
		
		//set up the block size
		MRJobConfiguration.setBlocksSizes(job, realIndexes, brlens, bclens);
		
		//set up unary instructions that will perform in the mapper
		MRJobConfiguration.setInstructionsInMapper(job, instructionsInMapper);
		
		//set up the aggregate instructions that will happen in the combiner and reducer
		MRJobConfiguration.setReblockInstructions(job, reblockInstructions);
		
		//set up the instructions that will happen in the reducer, after the aggregation instrucions
		MRJobConfiguration.setInstructionsInReducer(job, otherInstructionsInReducer);
		 
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);

		//disable automatic tasks timeouts and speculative task exec
		job.setInt("mapred.task.timeout", 0);			
		job.setMapSpeculativeExecution(false);
		
		//set up preferred custom serialization framework for binary block format
		if( MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION )
			MRJobConfiguration.addBinaryBlockSerializationFramework( job );
		
		//enable jvm reuse (based on SystemML configuration)
		if( jvmReuse )
			job.setNumTasksToExecutePerJvm(-1);
		
		//set up what matrices are needed to pass from the mapper to reducer
		HashSet<Byte> mapoutputIndexes=MRJobConfiguration.setUpOutputIndexesForMapper(job, realIndexes,  instructionsInMapper, 
				reblockInstructions, null, otherInstructionsInReducer, resultIndexes);
		
		MatrixChar_N_ReducerGroups ret=MRJobConfiguration.computeMatrixCharacteristics(job, realIndexes, 
				instructionsInMapper, reblockInstructions, null, null, otherInstructionsInReducer, 
				resultIndexes, mapoutputIndexes, false);
		
		MatrixCharacteristics[] stats=ret.stats;
		
		//set up the number of reducers (according to output size)
		int numRed = determineNumReducers(rlens, clens, nnz, ConfigurationManager.getConfig().getIntValue(DMLConfig.NUM_REDUCERS), (int)ret.numReducerGroups);
		job.setNumReduceTasks(numRed);
		
		//setup in-memory reduce buffers budget (reblock reducer dont need much memory)
		//job.set("mapred.job.reduce.input.buffer.percent", "0.70");
		
		// Print the complete instruction
		if (LOG.isTraceEnabled())
			inst.printCompleteMRJobInstruction(stats);
		
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
		job.setMapperClass(ReblockMapper.class);
		job.setMapOutputKeyClass(MatrixIndexes.class); //represent key offsets for block
		job.setMapOutputValueClass(TaggedAdaptivePartialBlock.class); //binary cell/block
		
		//configure reducer
		job.setReducerClass(ReblockReducer.class);
		
		
		// By default, the job executes in "cluster" mode.
		// Determine if we can optimize and run it in "local" mode.
		
		// at this point, both reblock_binary and reblock_text are similar
		MatrixCharacteristics[] inputStats = new MatrixCharacteristics[inputs.length];
		for ( int i=0; i < inputs.length; i++ ) {
			inputStats[i] = new MatrixCharacteristics(rlens[i], clens[i], brlens[i], bclens[i]);
		}
		ExecMode mode = RunMRJobs.getExecMode(JobType.REBLOCK, inputStats); 
		if ( mode == ExecMode.LOCAL ) {
			job.set("mapred.job.tracker", "local");
			MRJobConfiguration.setStagingDir( job );
		}
		
		//set unique working dir
		MRJobConfiguration.setUniqueWorkingDir(job, mode);
		
	
		RunningJob runjob=JobClient.runJob(job);
		
		/* Process different counters */
		
		Group group=runjob.getCounters().getGroup(MRJobConfiguration.NUM_NONZERO_CELLS);
		for(int i=0; i<resultIndexes.length; i++) {
			// number of non-zeros
			stats[i].nonZero=group.getCounter(Integer.toString(i));
			//	System.out.println("result #"+resultIndexes[i]+" ===>\n"+stats[i]);
		}

		return new JobReturn(stats, outputInfos, runjob.isSuccessful());
	}
	
	/**
	 * 
	 * @param rlen
	 * @param clen
	 * @param nnz
	 * @param defaultNumRed
	 * @param numRedGroups
	 * @return
	 */
	private static int determineNumReducers( long[] rlen, long[] clen, long[] nnz, int defaultNumRed, int numRedGroups )
	{
		//init return with default value
		int ret = defaultNumRed;
		
		//determine max output matrix size
		long maxNumRed = InfrastructureAnalyzer.getRemoteParallelReduceTasks();
		long blockSize = InfrastructureAnalyzer.getHDFSBlockSize()/(1024*1024);
		long maxSize = -1; //in MB
		for( int i=0; i<rlen.length; i++ )
		{			
			long tmp = MatrixBlock.estimateSizeOnDisk(rlen[i], clen[i], nnz[i]) / (1024*1024);
			maxSize = Math.max(maxSize, tmp);
		}
		
		//correction max number of reducers on yarn clusters
		if( InfrastructureAnalyzer.isYarnEnabled() )
			maxNumRed = Math.max( maxNumRed, YarnClusterAnalyzer.getNumCores()/2 );
		
		//increase num reducers wrt input size / hdfs blocksize (up to max reducers)
		ret = (int)Math.max(ret, Math.min(maxSize/blockSize, maxNumRed));
		
		//reduce num reducers for few result blocks
		ret = Math.min(ret, numRedGroups);
		
		//ensure there is at least one reducer
		ret = Math.max(ret, 1);
		
		return ret;
	}
}
