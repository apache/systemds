/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */


package org.apache.sysml.runtime.matrix;

import java.util.HashSet;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;

import org.apache.sysml.runtime.instructions.MRJobInstruction;
import org.apache.sysml.runtime.matrix.data.CM_N_COVCell;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.data.TaggedFirstSecondIndexes;
import org.apache.sysml.runtime.matrix.mapred.CMCOVMRMapper;
import org.apache.sysml.runtime.matrix.mapred.CMCOVMRReducer;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration.ConvertTarget;


public class CMCOVMR 
{
	private static final Log LOG = LogFactory.getLog(CMCOVMR.class.getName());
	
	private CMCOVMR() {
		//prevent instantiation via private constructor
	}
	
	public static JobReturn runJob(MRJobInstruction inst, String[] inputs, InputInfo[] inputInfos, long[] rlens, long[] clens, 
			int[] brlens, int[] bclens, String instructionsInMapper, String cmNcomInstructions, 
			int numReducers, int replication, byte[] resultIndexes,	String[] outputs, OutputInfo[] outputInfos) 
	throws Exception
	{
		JobConf job = new JobConf(CMCOVMR.class);
		job.setJobName("CM-COV-MR");
		
		//whether use block representation or cell representation
		MRJobConfiguration.setMatrixValueClassForCM_N_COM(job, true);
	
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
		MRJobConfiguration.setUpMultipleInputs(job, realIndexes, realinputs, realinputInfos, realbrlens, realbclens, true, ConvertTarget.WEIGHTEDCELL);
		
		//set up the dimensions of input matrices
		MRJobConfiguration.setMatricesDimensions(job, realIndexes, realrlens, realclens);
		
		//set up the block size
		MRJobConfiguration.setBlocksSizes(job, realIndexes, realbrlens, realbclens);
		
		//set up unary instructions that will perform in the mapper
		MRJobConfiguration.setInstructionsInMapper(job, instructionsInMapper);
		
		//set up the aggregate instructions that will happen in the combiner and reducer
		MRJobConfiguration.setCM_N_COMInstructions(job, cmNcomInstructions);
		
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);
		
		//set up what matrices are needed to pass from the mapper to reducer
		HashSet<Byte> mapoutputIndexes=MRJobConfiguration.setUpOutputIndexesForMapper(job, realIndexes, instructionsInMapper, null, 
				cmNcomInstructions, resultIndexes);
		
		//set up the multiple output files, and their format information
		MRJobConfiguration.setUpMultipleOutputs(job, resultIndexes, new byte[resultIndexes.length], outputs, outputInfos, false);
		
		// configure mapper and the mapper output key value pairs
		job.setMapperClass(CMCOVMRMapper.class);
		
		job.setMapOutputKeyClass(TaggedFirstSecondIndexes.class);
		job.setMapOutputValueClass(CM_N_COVCell.class);
		job.setOutputKeyComparatorClass(TaggedFirstSecondIndexes.Comparator.class);
		job.setPartitionerClass(TaggedFirstSecondIndexes.TagPartitioner.class);
		
		//configure reducer
		job.setReducerClass(CMCOVMRReducer.class);
		//job.setReducerClass(PassThroughReducer.class);
		
		MatrixCharacteristics[] stats=MRJobConfiguration.computeMatrixCharacteristics(job, realIndexes, 
				instructionsInMapper, null, null, cmNcomInstructions, resultIndexes, mapoutputIndexes, false).stats;
		
		//set up the number of reducers
		MRJobConfiguration.setNumReducers(job, mapoutputIndexes.size(), numReducers);//each output tag is a group
		
		// Print the complete instruction
		if (LOG.isTraceEnabled())
			inst.printCompleteMRJobInstruction(stats);
		
		
		// By default, the job executes in "cluster" mode.
		// Determine if we can optimize and run it in "local" mode.
		MatrixCharacteristics[] inputStats = new MatrixCharacteristics[inputs.length];
		for ( int i=0; i < inputs.length; i++ ) {
			inputStats[i] = new MatrixCharacteristics(rlens[i], clens[i], brlens[i], bclens[i]);
		}
		
		//set unique working dir
		MRJobConfiguration.setUniqueWorkingDir(job);
		
		
		RunningJob runjob=JobClient.runJob(job);
		
		return new JobReturn(stats, outputInfos, runjob.isSuccessful());
	}

}
