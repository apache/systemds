/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */


package org.apache.sysml.runtime.matrix;

import java.util.HashSet;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.mapred.Counters.Group;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.instructions.MRInstructionParser;
import org.apache.sysml.runtime.instructions.MRJobInstruction;
import org.apache.sysml.runtime.instructions.mr.AggregateBinaryInstruction;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.data.TaggedFirstSecondIndexes;
import org.apache.sysml.runtime.matrix.mapred.MMCJMRMapper;
import org.apache.sysml.runtime.matrix.mapred.MMCJMRReducerWithAggregator;
import org.apache.sysml.runtime.matrix.mapred.MRConfigurationNames;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration.ConvertTarget;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration.MatrixChar_N_ReducerGroups;
import org.apache.sysml.yarn.DMLAppMasterUtils;
import org.apache.sysml.yarn.ropt.YarnClusterAnalyzer;


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
	private static final boolean AUTOMATIC_CONFIG_NUM_REDUCERS = true;
	private static final Log LOG = LogFactory.getLog(MMCJMR.class);

	private MMCJMR() {
		//prevent instantiation via private constructor
	}
	
	public static JobReturn runJob(MRJobInstruction inst, String[] inputs, InputInfo[] inputInfos, long[] rlens, long[] clens, 
			int[] brlens, int[] bclens, String instructionsInMapper, 
			String aggInstructionsInReducer, String aggBinInstrction, int numReducers, 
			int replication, String output, OutputInfo outputinfo) 
	throws Exception
	{
		JobConf job = new JobConf(MMCJMR.class);
		
		// TODO: check w/ yuanyuan. This job always runs in blocked mode, and hence derivation is not necessary.
		boolean inBlockRepresentation=MRJobConfiguration.deriveRepresentation(inputInfos);
		
		// by default, assume that dimensions of MMCJ's output are known at compile time
		byte resultDimsUnknown = (byte) 0;   
		MatrixCharacteristics[] stats=commonSetup(job, inBlockRepresentation, inputs, inputInfos, rlens, clens, 
				brlens, bclens, instructionsInMapper, aggInstructionsInReducer, aggBinInstrction, numReducers, 
				replication, resultDimsUnknown, output, outputinfo);
		
		// Print the complete instruction
		if (LOG.isTraceEnabled())
			inst.printCompleteMRJobInstruction(stats);
		
		// Update resultDimsUnknown based on computed "stats"
		// There is always a single output
		if ( stats[0].getRows() == -1 || stats[0].getCols() == -1 ) {
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
		
		if(dim1.getRowsPerBlock()>dim1.getRows())
			dim1.setRowsPerBlock( (int) dim1.getRows() );
		if(dim1.getColsPerBlock()>dim1.getCols())
			dim1.setColsPerBlock( (int) dim1.getCols() );
		if(dim2.getRowsPerBlock()>dim2.getRows())
			dim2.setRowsPerBlock( (int) dim2.getRows() );
		if(dim2.getColsPerBlock()>dim2.getCols())
			dim2.setColsPerBlock( (int) dim2.getCols() );
	
		long blockSize1=77+8*dim1.getRowsPerBlock()*dim1.getColsPerBlock();
		long blockSize2=77+8*dim2.getRowsPerBlock()*dim2.getColsPerBlock();
		long blockSizeResult=77+8*dim1.getRowsPerBlock()*dim2.getColsPerBlock();
		
		long cacheSize = -1;
		//cache the first result
		if(dim1.getRows()<dim2.getCols())
		{
			long numBlocks=(long)Math.ceil((double)dim1.getRows()/(double)dim1.getRowsPerBlock());
			cacheSize=numBlocks*(20+blockSize1)+32;
		}
		else //cache the second result
		{
			long numBlocks=(long)Math.ceil((double)dim2.getCols()/(double) dim2.getColsPerBlock());
			cacheSize=numBlocks*(20+blockSize2)+32;
		}
		//add known memory consumption (will be substracted from output buffer)
		cacheSize += 2* Math.max(blockSize1, blockSize2) //the cached key-value pair  (plus input instance)
				  +  blockSizeResult //the cached single result
				  +  MRJobConfiguration.getMiscMemRequired(job); //misc memory requirement by hadoop
		MRJobConfiguration.setMMCJCacheSize(job, (int)cacheSize);
		
		//set unique working dir
		MRJobConfiguration.setUniqueWorkingDir(job); 
		
		//run mmcj job
		RunningJob runjob=JobClient.runJob(job);
		
		/* Process different counters */
		
		// NOTE: MMCJ job always has only a single output. 
		// Hence, no need to scan resultIndexes[] like other jobs
		
		int outputIndex = 0;
		Byte outputMatrixID = MRInstructionParser.parseSingleInstruction(aggBinInstrction).output;
		
		Group group=runjob.getCounters().getGroup(MRJobConfiguration.NUM_NONZERO_CELLS);
		
		// number of non-zeros
		stats[outputIndex].setNonZeros(group.getCounter(Byte.toString(outputMatrixID)));
		
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
		job.setInt(MRConfigurationNames.DFS_REPLICATION, replication);

		//set up preferred custom serialization framework for binary block format
		if( MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION )
			MRJobConfiguration.addBinaryBlockSerializationFramework( job );
		
		//set up map/reduce memory configurations (if in AM context)
		DMLConfig config = ConfigurationManager.getConfig();
		DMLAppMasterUtils.setupMRJobRemoteMaxMemory(job, config);

		//set up custom map/reduce configurations 
		MRJobConfiguration.setupCustomMRConfigurations(job, config);
		
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
		
	    //if(aggInstructionsInReducer!=null && !aggInstructionsInReducer.isEmpty())
	    //	job.setCombinerClass(MMCJMRCombiner.class);
		
		MatrixChar_N_ReducerGroups ret=MRJobConfiguration.computeMatrixCharacteristics(job, realIndexes, 
				instructionsInMapper, aggInstructionsInReducer, aggBinInstrction, null, resultIndexes, 
				mapoutputIndexes, true);
		
		//set up the number of reducers
		if( AUTOMATIC_CONFIG_NUM_REDUCERS ){
			int numRed = determineNumReducers(rlens, clens, numReducers, ret.numReducerGroups);
			job.setNumReduceTasks(numRed);
		}
		else
			MRJobConfiguration.setNumReducers(job, ret.numReducerGroups, numReducers);

		//configure reducer
		// note: the alternative MMCJMRReducer is not maintained
		job.setReducerClass(MMCJMRReducerWithAggregator.class);
		
		return ret.stats;
	}
	
	/**
	 * Determine number of reducers based on configured number of reducers, number of results groups
	 * and input data divided by blocksize (as heuristic for useful degree of parallelism).
	 * 
	 * @param rlen
	 * @param clen
	 * @param defaultNumRed
	 * @param numRedGroups
	 * @return
	 */
	protected static int determineNumReducers( long[] rlen, long[] clen, int defaultNumRed, long numRedGroups )
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
		//as a heuristic we allow an increase up to 2x the configured default, now disabled
		//maxNumRed = Math.min(2 * defaultNumRed, maxNumRed);
		ret = (int)Math.max(ret, Math.min(maxSize/blockSize, maxNumRed));
		
		//reduce num reducers for few result blocks
		ret = (int) Math.min(ret, numRedGroups);
		
		//ensure there is at least one reducer
		ret = Math.max(ret, 1);
		
		return ret;
	}
}
