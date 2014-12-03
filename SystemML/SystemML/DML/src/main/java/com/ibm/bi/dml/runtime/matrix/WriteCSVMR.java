/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix;

import java.util.HashMap;
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
import com.ibm.bi.dml.runtime.instructions.MRInstructionParser;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CSVWriteInstruction;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.data.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.CSVWriteMapper;
import com.ibm.bi.dml.runtime.matrix.mapred.CSVWriteReducer;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration.ConvertTarget;
import com.ibm.bi.dml.yarn.ropt.YarnClusterAnalyzer;

public class WriteCSVMR 
{
    @SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		

	private static final Log LOG = LogFactory.getLog(WriteCSVMR.class.getName());
	
	public static JobReturn runJob(MRJobInstruction inst, String[] inputs, InputInfo[] inputInfos, 
			long[] rlens, long[] clens, int[] brlens, int[] bclens, String csvWriteInstructions, int numReducers, int replication, 
			byte[] resultIndexes, String[] outputs) 
	throws Exception
	{
		//assert(inputs.length==outputs.length);
		JobConf job;
		job = new JobConf(WriteCSVMR.class);
		job.setJobName("WriteCSV-MR");
		
		byte[] realIndexes=new byte[inputs.length];
		for(byte b=0; b<realIndexes.length; b++)
			realIndexes[b]=b;
		
		//set up the input files and their format information
		MRJobConfiguration.setUpMultipleInputs(job, realIndexes, inputs, inputInfos, brlens, bclens, true, ConvertTarget.CSVWRITE);
		
		//set up the dimensions of input matrices
		MRJobConfiguration.setMatricesDimensions(job, realIndexes, rlens, clens);
		
		//set up the block size
		MRJobConfiguration.setBlocksSizes(job, realIndexes, brlens, bclens);
		
		MRJobConfiguration.setCSVWriteInstructions(job, csvWriteInstructions);
		
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);
		
		//set up preferred custom serialization framework for binary block format
		if( MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION )
			MRJobConfiguration.addBinaryBlockSerializationFramework( job );
		
		long maxRlen=0;
		for(long rlen: rlens)
			if(rlen>maxRlen)
				maxRlen=rlen;
		
		//set up the number of reducers (according to output size)
		int numRed = determineNumReducers(rlens, clens, ConfigurationManager.getConfig().getIntValue(DMLConfig.NUM_REDUCERS), (int)maxRlen);
		job.setNumReduceTasks(numRed);
		
		byte[] resultDimsUnknown = new byte[resultIndexes.length];
		MatrixCharacteristics[] stats=new MatrixCharacteristics[resultIndexes.length];
		OutputInfo[] outputInfos=new OutputInfo[outputs.length];
		HashMap<Byte, Integer> indexmap=new HashMap<Byte, Integer>();
		for(int i=0; i<stats.length; i++)
		{
			indexmap.put(resultIndexes[i], i);
			resultDimsUnknown[i] = (byte) 0;
			stats[i]=new MatrixCharacteristics();
			outputInfos[i]=OutputInfo.CSVOutputInfo;
		}
		CSVWriteInstruction[] ins = MRInstructionParser.parseCSVWriteInstructions(csvWriteInstructions);
		for(CSVWriteInstruction in: ins)
			stats[indexmap.get(in.output)].set(rlens[in.input], clens[in.input], -1, -1);
		
		// Print the complete instruction
		if (LOG.isTraceEnabled())
			inst.printCompleteMRJobInstruction(stats);
		
		//set up what matrices are needed to pass from the mapper to reducer
		HashSet<Byte> mapoutputIndexes=MRJobConfiguration.setUpOutputIndexesForMapper(job, realIndexes,  "", 
				"", csvWriteInstructions, resultIndexes);
		
		//set up the multiple output files, and their format information
		MRJobConfiguration.setUpMultipleOutputs(job, resultIndexes, resultDimsUnknown, outputs, 
				outputInfos, true, true);
		
		// configure mapper and the mapper output key value pairs
		job.setMapperClass(CSVWriteMapper.class);
		job.setMapOutputKeyClass(TaggedFirstSecondIndexes.class);
		job.setMapOutputValueClass(MatrixBlock.class);
		
		//configure reducer
		job.setReducerClass(CSVWriteReducer.class);
		job.setOutputKeyComparatorClass(TaggedFirstSecondIndexes.Comparator.class);
		job.setPartitionerClass(TaggedFirstSecondIndexes.FirstIndexRangePartitioner.class);
		//job.setOutputFormat(UnPaddedOutputFormat.class);
	
		
		MatrixCharacteristics[] inputStats = new MatrixCharacteristics[inputs.length];
		for ( int i=0; i < inputs.length; i++ ) {
			inputStats[i] = new MatrixCharacteristics(rlens[i], clens[i], brlens[i], bclens[i]);
		}
		
		// By default, the job executes in "cluster" mode.
		// Determine if we can optimize and run it in "local" mode.
				
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
	 * @param defaultNumRed
	 * @param numRedGroups
	 * @return
	 */
	protected static int determineNumReducers( long[] rlen, long[] clen, int defaultNumRed, int numRedGroups )
	{
		//init return with default value
		int ret = defaultNumRed;
		
		//determine max output matrix size
		long maxNumRed = InfrastructureAnalyzer.getRemoteParallelReduceTasks();
		long blockSize = InfrastructureAnalyzer.getHDFSBlockSize()/(1024*1024);
		long maxSize = -1; //in MB
		for( int i=0; i<rlen.length; i++ )
		{			
			long tmp = MatrixBlock.estimateSizeOnDisk(rlen[i], clen[i], rlen[i]*clen[i]) / (1024*1024);
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
