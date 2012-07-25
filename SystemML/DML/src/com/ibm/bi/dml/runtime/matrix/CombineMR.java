package com.ibm.bi.dml.runtime.matrix;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Vector;

import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.RunningJob;

import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.lops.runtime.RunMRJobs;
import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CombineBinaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CombineTertiaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MRInstruction;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.Pair;
import com.ibm.bi.dml.runtime.matrix.io.TaggedMatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.TaggedMatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.TaggedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.WeightedPair;
import com.ibm.bi.dml.runtime.matrix.mapred.GMRMapper;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.matrix.mapred.ReduceBase;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.bi.dml.utils.configuration.DMLConfig;


public class CombineMR {
	
	public static class InnerReducer extends ReduceBase
	implements Reducer<MatrixIndexes, TaggedMatrixValue, MatrixIndexes, WeightedPair>
	{

		protected MRInstruction[] comb_instructions=null;
		private MatrixIndexes keyBuff=new MatrixIndexes();
		private WeightedPair valueBuff=new WeightedPair();
		private HashMap<Byte, Pair<Integer, Integer>> outputBlockSizes=new HashMap<Byte, Pair<Integer, Integer>>();
		private HashMap<Byte, Vector<Integer>> outputIndexesMapping=new HashMap<Byte, Vector<Integer>>();
		@Override
		public void reduce(MatrixIndexes indexes,
				Iterator<TaggedMatrixValue> values,
				OutputCollector<MatrixIndexes, WeightedPair> out, Reporter reporter)
				throws IOException {
			
			long start=System.currentTimeMillis();
			
			if(firsttime)
			{
				cachedReporter=reporter;
				firsttime=false;
			}
			
			cachedValues.reset();
			
			while(values.hasNext())
			{
				TaggedMatrixValue taggedValue=values.next();
				cachedValues.set(taggedValue.getTag(), indexes, taggedValue.getBaseObject());
			}
			//LOG.info("before aggregation: \n"+cachedValues);
			//perform aggregate operations first
			//processAggregateInstructions(indexes, values);
			
			//LOG.info("after aggregation: \n"+cachedValues);
			
			//perform mixed operations
			//processReducerInstructions();
			
			processCombineInstructionsAndOutput(reporter);

			reporter.incrCounter(Counters.COMBINE_OR_REDUCE_TIME, System.currentTimeMillis()-start);
			
		}
		
		public void configure(JobConf job)
		{	
			super.configure(job);
			try {
				comb_instructions = MRJobConfiguration.getCombineInstruction(job);
				
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			for(int i=0; i<resultIndexes.length; i++)
			{
				MatrixCharacteristics stat=MRJobConfiguration.getMatrixCharacteristicsForOutput(job, resultIndexes[i]);
				outputBlockSizes.put(resultIndexes[i], new Pair<Integer, Integer>(stat.numRowsPerBlock, stat.numColumnsPerBlock));
			}
			for(MRInstruction ins: comb_instructions)
			{
				outputIndexesMapping.put(ins.output, getOutputIndexes(ins.output));
			}
		}
		
		void processCombineInstructionsAndOutput(Reporter reporter) 
		throws IOException
		{
			for(MRInstruction ins: comb_instructions)
			{
				if(ins instanceof CombineBinaryInstruction)
					processBinaryCombineInstruction((CombineBinaryInstruction)ins, reporter);
				else if(ins instanceof CombineTertiaryInstruction)
					processTertiaryCombineInstruction((CombineTertiaryInstruction)ins, reporter);
				else
					throw new IOException("unsupported instruction: "+ins);
			}
		}

		private void processTertiaryCombineInstruction(
				CombineTertiaryInstruction ins, Reporter reporter) throws IOException{
			IndexedMatrixValue in1=cachedValues.getFirst(ins.input1);
			IndexedMatrixValue in2=cachedValues.getFirst(ins.input2);
			IndexedMatrixValue in3=cachedValues.getFirst(ins.input3);
			if(in1==null && in2==null && in3==null)
				return;
			int nr=0, nc=0;
			MatrixIndexes indexes=null;
			if(in1!=null)
			{
				nr=in1.getValue().getNumRows();
				nc=in1.getValue().getNumColumns();
				indexes=in1.getIndexes();
			}else if(in2!=null)
			{
				nr=in2.getValue().getNumRows();
				nc=in2.getValue().getNumColumns();
				indexes=in2.getIndexes();
			}else
			{
				nr=in3.getValue().getNumRows();
				nc=in3.getValue().getNumColumns();
				indexes=in3.getIndexes();
			}
			
			//if one of the inputs is null, then it is a all zero block
			if(in1==null)
			{
				in1=zeroInput;
				in1.getValue().reset(nr, nc);
			}
			
			if(in2==null)
			{
				in2=zeroInput;
				in2.getValue().reset(nr, nc);
			}
			
			if(in3==null)
			{
				in3=zeroInput;
				in3.getValue().reset(nr, nc);
			}
			
			
			//process instruction
			try {
				
				Vector<Integer> outputIndexes = outputIndexesMapping.get(ins.output);
				for(int r=0; r<nr; r++)
					for(int c=0; c<nc; c++)
					{
						Pair<Integer, Integer> blockSize=outputBlockSizes.get(ins.output);
						keyBuff.setIndexes(
								UtilFunctions.cellIndexCalculation(in1.getIndexes().getRowIndex(), blockSize.getKey(), r),
								UtilFunctions.cellIndexCalculation(in1.getIndexes().getColumnIndex(), blockSize.getValue(), c)
								);
						valueBuff.setValue(in1.getValue().getValue(r, c));
						valueBuff.setOtherValue(in2.getValue().getValue(r, c));
						valueBuff.setWeight(in3.getValue().getValue(r, c));
						for(int i: outputIndexes)
						{
							collectFinalMultipleOutputs.collectOutput(keyBuff, valueBuff, i, reporter);
							//System.out.println("output: "+keyBuff+" -- "+valueBuff);
						}
					}
				
			} catch (Exception e) {
				throw new RuntimeException(e);
			} 
			
		}

		private void processBinaryCombineInstruction(CombineBinaryInstruction ins, Reporter reporter) 
		throws IOException 
		{
			
			IndexedMatrixValue in1=cachedValues.getFirst(ins.input1);
			IndexedMatrixValue in2=cachedValues.getFirst(ins.input2);
			if(in1==null && in2==null)
				return;
			
			MatrixIndexes indexes;
			if(in1!=null)
				indexes=in1.getIndexes();
			else
				indexes=in2.getIndexes();
			
			//if one of the inputs is null, then it is a all zero block
			if(in1==null)
			{
				in1=zeroInput;
				in1.getValue().reset(in2.getValue().getNumRows(), 
						in2.getValue().getNumColumns());
			}
			
			if(in2==null)
			{
				in2=zeroInput;
				in2.getValue().reset(in1.getValue().getNumRows(), 
						in1.getValue().getNumColumns());
			}
			
			//System.out.println("in1:"+in1);
			//System.out.println("in2:"+in2);
			
			//process instruction
			try {
				/*in1.getValue().combineOperations(in2.getValue(), collectFinalMultipleOutputs, 
						reporter, keyBuff, valueBuff, getOutputIndexes(ins.output));*/
				
				Vector<Integer> outputIndexes = outputIndexesMapping.get(ins.output);
				for(int r=0; r<in1.getValue().getNumRows(); r++)
					for(int c=0; c<in1.getValue().getNumColumns(); c++)
					{
						Pair<Integer, Integer> blockSize=outputBlockSizes.get(ins.output);
						keyBuff.setIndexes(
								UtilFunctions.cellIndexCalculation(indexes.getRowIndex(), blockSize.getKey(), r),
								UtilFunctions.cellIndexCalculation(indexes.getColumnIndex(), blockSize.getValue(), c)
								);
						valueBuff.setValue(in1.getValue().getValue(r, c));
						double temp=in2.getValue().getValue(r, c);
						if(ins.isSecondInputWeight())
						{
							valueBuff.setWeight(temp);
							valueBuff.setOtherValue(0);
						}
						else
						{
							valueBuff.setWeight(1);
							valueBuff.setOtherValue(temp);
						}
						
						for(int i: outputIndexes)
						{
							collectFinalMultipleOutputs.collectOutput(keyBuff, valueBuff, i, reporter);
							//System.out.println("output: "+keyBuff+" -- "+valueBuff);
						}
					}
				
			} catch (Exception e) {
				throw new RuntimeException(e);
			} 
		}

	}

	@SuppressWarnings("deprecation")
	public static JobReturn runJob(String[] inputs, InputInfo[] inputInfos, 
			long[] rlens, long[] clens, int[] brlens, int[] bclens, String combineInstructions, 
			int numReducers, int replication, byte[] resultIndexes, String[] outputs, OutputInfo[] outputInfos) 
	throws Exception
	{
		boolean inBlockRepresentation=MRJobConfiguration.deriveRepresentation(inputInfos);
		return runJob(inBlockRepresentation, inputs, inputInfos, 
				rlens, clens, brlens, bclens, combineInstructions, 
				numReducers, replication, resultIndexes, outputs, outputInfos);
	}
	@SuppressWarnings("deprecation")
	public static JobReturn runJob(boolean inBlockRepresentation, String[] inputs, InputInfo[] inputInfos, 
			long[] rlens, long[] clens, int[] brlens, int[] bclens, String combineInstructions, 
			int numReducers, int replication, byte[] resultIndexes, String[] outputs, OutputInfo[] outputInfos) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(CombineMR.class);
		job.setJobName("Standalone-MR");
		
		//whether use block representation or cell representation
		MRJobConfiguration.setMatrixValueClass(job, inBlockRepresentation);
		
		byte[] inputIndexes=new byte[inputs.length];
		for(byte b=0; b<inputs.length; b++)
			inputIndexes[b]=b;
		
		//set up the input files and their format information
		MRJobConfiguration.setUpMultipleInputs(job, inputIndexes, inputs, inputInfos, inBlockRepresentation, brlens, bclens);
		
		//set up the dimensions of input matrices
		MRJobConfiguration.setMatricesDimensions(job, inputIndexes, rlens, clens);
		
		//set up the block size
		MRJobConfiguration.setBlocksSizes(job, inputIndexes, brlens, bclens);
		
		//set up unary instructions that will perform in the mapper
		MRJobConfiguration.setInstructionsInMapper(job, "");
		
		//set up the aggregate instructions that will happen in the combiner and reducer
		MRJobConfiguration.setAggregateInstructions(job, "");
		
		//set up the instructions that will happen in the reducer, after the aggregation instrucions
		MRJobConfiguration.setInstructionsInReducer(job, "");
		
		MRJobConfiguration.setCombineInstructions(job, combineInstructions);
		
		//set up the number of reducers
		job.setNumReduceTasks(numReducers);
		
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);
		
		//set up what matrices are needed to pass from the mapper to reducer
		MRJobConfiguration.setUpOutputIndexesForMapper(job, inputIndexes, null, null, combineInstructions, 
				resultIndexes);
		
		//set up the multiple output files, and their format information
		MRJobConfiguration.setUpMultipleOutputs(job, resultIndexes, null, outputs, outputInfos, inBlockRepresentation);
		
		// configure mapper and the mapper output key value pairs
		job.setMapperClass(GMRMapper.class);
		
		job.setMapOutputKeyClass(MatrixIndexes.class);
		if(inBlockRepresentation)
			job.setMapOutputValueClass(TaggedMatrixBlock.class);
		else
			job.setMapOutputValueClass(TaggedMatrixCell.class);
	
		//configure reducer
		job.setReducerClass(InnerReducer.class);
		//job.setReducerClass(PassThroughReducer.class);
		
		
		MatrixCharacteristics[] stats=MRJobConfiguration.computeMatrixCharacteristics(job, inputIndexes,  
				null, null, null, combineInstructions, resultIndexes);
		
		// By default, the job executes in "cluster" mode.
		// Determine if we can optimize and run it in "local" mode.
		MatrixCharacteristics[] inputStats = new MatrixCharacteristics[inputs.length];
		for ( int i=0; i < inputs.length; i++ ) {
			inputStats[i] = new MatrixCharacteristics(rlens[i], clens[i], brlens[i], bclens[i]);
		}
		ExecMode mode = RunMRJobs.getExecMode(JobType.COMBINE, inputStats); 
		if ( mode == ExecMode.LOCAL ) {
			job.set("mapred.job.tracker", "local");
			job.set("mapreduce.jobtracker.staging.root.dir", DMLConfig.LOCAL_MR_MODE_STAGING_DIR);
		}

		//set unique working dir
		MRJobConfiguration.setUniqueWorkingDir(job, mode);
		
		
		RunningJob runjob=JobClient.runJob(job);

	    return new JobReturn(stats, runjob.isSuccessful());
	}
}
