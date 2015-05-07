/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.instructions.mr.AggregateInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.MRInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.TernaryInstruction;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.data.TaggedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.util.MapReduceTool;


public class ReduceBase extends MRBaseForCommonInstructions
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	//aggregate instructions
	protected HashMap<Byte, ArrayList<AggregateInstruction>> 
	agg_instructions=new HashMap<Byte, ArrayList<AggregateInstruction>>();
	
	//default aggregate operation
	protected static final AggregateOperator DEFAULT_AGG_OP = new AggregateOperator(0, Plus.getPlusFnObject());

	//default aggregate instruction
	protected AggregateInstruction defaultAggIns=
		new AggregateInstruction(DEFAULT_AGG_OP, (byte)0, (byte)0, "DEFAULT_AGG_OP");
	
	//mixsure of instructions performed in reducer
	protected ArrayList<MRInstruction> mixed_instructions = null;
	
	//the final result indexes that needed to be outputted
	protected byte[] resultIndexes=null;
	protected byte[] resultDimsUnknown=null;

	//output converters
	protected CollectMultipleConvertedOutputs collectFinalMultipleOutputs;
	
	//a counter to calculate the time spent in a reducer or a combiner
	public static enum Counters {COMBINE_OR_REDUCE_TIME };

	//the counters to record how many nonZero cells have been produced for each output
	protected long[] resultsNonZeros=null;
	protected long[] resultsMaxRowDims=null;
	protected long[] resultsMaxColDims=null;
	protected String dimsUnknownFilePrefix;
	
	//cached reporter to report the number of nonZeros for each reduce task
	protected Reporter cachedReporter=null;
	protected boolean firsttime=true;
	
	protected String reducerID;
	
	//just for stable aggregation function
	//a cache to hold the corrections for aggregation results
	protected CachedValueMap correctionCache=new CachedValueMap();
	
	protected void commonSetup(Reporter reporter)
	{
		if(firsttime)
		{
			cachedReporter=reporter;
			firsttime=false;
		}
	}
	
	public void configure(JobConf job)
	{	
		super.configure(job);
		
		reducerID = job.get("mapred.task.id");
		dimsUnknownFilePrefix = job.get("dims.unknown.file.prefix");

		
		//get the indexes of the final output matrices
		resultIndexes=MRJobConfiguration.getResultIndexes(job);
		resultDimsUnknown = MRJobConfiguration.getResultDimsUnknown(job);
		
		//initialize SystemML Counters (defined in MRJobConfiguration)
		resultsNonZeros=new long[resultIndexes.length];
		resultsMaxRowDims=new long[resultIndexes.length];
		resultsMaxColDims=new long[resultIndexes.length];
		
		collectFinalMultipleOutputs = MRJobConfiguration.getMultipleConvertedOutputs(job);
		
		//parse aggregate operations
		AggregateInstruction[] agg_insts=null;
		try {
			agg_insts = MRJobConfiguration.getAggregateInstructions(job);
			//parse unary and binary operations
			MRInstruction[] tmp = MRJobConfiguration.getInstructionsInReducer(job);
			if( tmp != null ) {
				mixed_instructions=new ArrayList<MRInstruction>();
				Collections.addAll(mixed_instructions, tmp);
			}
			
			
		} catch (DMLUnsupportedOperationException e) {
			throw new RuntimeException(e);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}		
		
		//load data from distributed cache (if required, reuse if jvm_reuse)
		try {
			setupDistCacheFiles(job);
		}
		catch(IOException ex)
		{
			throw new RuntimeException(ex);
		}
		
		//reorganize the aggregate instructions, so that they are all associatied with each input
		if(agg_insts!=null)
		{
			for(AggregateInstruction ins: agg_insts)
			{
				//associate instruction to its input
				ArrayList<AggregateInstruction> vec=agg_instructions.get(ins.input);
				if(vec==null)
				{
					vec = new ArrayList<AggregateInstruction>();
					agg_instructions.put(ins.input, vec);
				}
				vec.add(ins);
				
				if(ins.input==ins.output)
					continue;
				
				//need to add new aggregate instructions so that partial aggregation can be applied
				//this is important for combiner in the reducer side
				AggregateInstruction partialIns=
					new AggregateInstruction(ins.getOperator(), ins.output, ins.output, ins.toString());
				vec=agg_instructions.get(partialIns.input);
				if(vec==null)
				{
					vec=new ArrayList<AggregateInstruction>();
					agg_instructions.put(partialIns.input, vec);
				}
				vec.add(partialIns);
			}
		}
	}
	
	protected void collectOutput_N_Increase_Counter(MatrixIndexes indexes, MatrixValue value, 
			int i, Reporter reporter) throws IOException
	{
		collectOutput_N_Increase_Counter(indexes, value, i, reporter, collectFinalMultipleOutputs, 
				resultDimsUnknown, resultsNonZeros, resultsMaxRowDims, resultsMaxColDims);
	}
	
	protected ArrayList<Integer> getOutputIndexes(byte outputTag)
	{
		ArrayList<Integer> ret = new ArrayList<Integer>();
		for(int i=0; i<resultIndexes.length; i++)
			if(resultIndexes[i]==outputTag)
				ret.add(i);
		return ret;
	}
	
	protected static ArrayList<Integer> getOutputIndexes(byte outputTag, byte[] resultIndexes)
	{
		ArrayList<Integer> ret=new ArrayList<Integer>();
		for(int i=0; i<resultIndexes.length; i++)
			if(resultIndexes[i]==outputTag)
				ret.add(i);
		return ret;
	}
	
	public void close() throws IOException
	{
		if(cachedReporter!=null)
		{
			String[] parts = reducerID.split("_");
			String jobID = "job_" + parts[1] + "_" + parts[2];
			int taskid;
			if ( parts[0].equalsIgnoreCase("task")) {
				taskid = Integer.parseInt(parts[parts.length-1]);
			}
			else if ( parts[0].equalsIgnoreCase("attempt")) {
				taskid = Integer.parseInt(parts[parts.length-2]);
			}
			else {
				throw new RuntimeException("Unrecognized format for reducerID: " + reducerID);
			}
			//System.out.println("Inside ReduceBase.close(): ID = " + reducerID + ", taskID = " + taskid);
			
			boolean dimsUnknown = false;
			for(int i=0; i<resultIndexes.length; i++) {
				cachedReporter.incrCounter(MRJobConfiguration.NUM_NONZERO_CELLS, Integer.toString(i), resultsNonZeros[i]);
				
				if ( resultDimsUnknown!=null && resultDimsUnknown[i] != (byte) 0 ) {
					dimsUnknown = true;
					// Each counter is of the form: (group, name)
					// where group = max_rowdim_resultindex; name = taskid
					//System.out.println("--> before i="+i+", row = " + cachedReporter.getCounter("max_rowdim_"+i, ""+taskid).getCounter() + ", col = " + cachedReporter.getCounter("max_coldim_"+i, ""+taskid).getCounter());
					//cachedReporter.getCounter(MRJobConfiguration.MAX_ROW_DIMENSION, Integer.toString(i)).increment(resultsMaxRowDims[i]);
					//cachedReporter.getCounter(MRJobConfiguration.MAX_COL_DIMENSION, Integer.toString(i)).increment(resultsMaxColDims[i]);
					//System.out.println("--> after i="+i+", row = " + cachedReporter.getCounter("max_rowdim_"+i, ""+taskid).getCounter() + ", col = " + cachedReporter.getCounter("max_coldim_"+i, ""+taskid).getCounter());
				}
			}
			
			//System.out.println("DimsUnknown = " + dimsUnknown);
			if ( dimsUnknown ) {
				// every task creates a file with max_row and max_col dimensions found in that task
		        MapReduceTool.writeDimsFile(dimsUnknownFilePrefix + "/" + jobID + "_dimsFile/" + "r_" + taskid , resultDimsUnknown, resultsMaxRowDims, resultsMaxColDims);
			}
		}
		collectFinalMultipleOutputs.close();
		
	}

	protected void processReducerInstructions() throws IOException
	{
		//perform mixed operations
		try {
			processMixedInstructions(mixed_instructions);
		} catch (Exception e) {
			throw new IOException(e);
		}
	}

	protected void outputInCombinerFromCachedValues(MatrixIndexes indexes, TaggedMatrixValue taggedbuffer, 
			OutputCollector<MatrixIndexes, TaggedMatrixValue> out) throws IOException
	{
		for(byte output: cachedValues.getIndexesOfAll())
		{
			ArrayList<IndexedMatrixValue> outValues=cachedValues.get(output);
			if(outValues==null)
				continue;
			for(IndexedMatrixValue outValue: outValues)
			{
				taggedbuffer.setBaseObject(outValue.getValue());
				taggedbuffer.setTag(output);
				out.collect(indexes, taggedbuffer);
				//System.out.println("**** combiner output: "+indexes+", "+taggedbuffer);
			}
		}
	}
	
	protected void outputResultsFromCachedValues(Reporter reporter) throws IOException
	{
		for(int i=0; i<resultIndexes.length; i++)
		{
			byte output=resultIndexes[i];
			ArrayList<IndexedMatrixValue> outValues=cachedValues.get(output);
			if(outValues==null)
				continue;
			for(IndexedMatrixValue outValue: outValues)
				collectOutput_N_Increase_Counter(outValue.getIndexes(), 
					outValue.getValue(), i, reporter);
	//		LOG.info("output: "+outValue.getIndexes()+" -- "+outValue.getValue()+" ~~ tag: "+output);
		//	System.out.println("Reducer output: "+outValue.getIndexes()+" -- "+outValue.getValue()+" ~~ tag: "+output);
		}
	}

	
	//process one aggregate instruction
/*	private void processAggregateHelp(MatrixIndexes indexes, TaggedMatrixValue tagged, 
			AggregateInstruction instruction) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		AggregateOperator aggOp=(AggregateOperator)instruction.getOperator();
		
		IndexedMatrixValue out=cachedValues.get(instruction.output);
		IndexedMatrixValue correction=null;
		if(aggOp.correctionExists)
			correction=correctionCache.get(instruction.output);
		if(out==null)
		{
			out=cachedValues.holdPlace(instruction.output, valueClass);
			out.getIndexes().setIndexes(indexes);
			if(aggOp.correctionExists )
			{
				if(correction==null)
					correction=correctionCache.holdPlace(instruction.output, valueClass);
				OperationsOnMatrixValues.startAggregation(out.getValue(), correction.getValue(), aggOp, 
					tagged.getBaseObject().getNumRows(), tagged.getBaseObject().getNumColumns(),
					tagged.getBaseObject().isInSparseFormat());
			}else
				OperationsOnMatrixValues.startAggregation(out.getValue(), null, aggOp, 
						tagged.getBaseObject().getNumRows(), tagged.getBaseObject().getNumColumns(),
						tagged.getBaseObject().isInSparseFormat());
		}
		
		if(aggOp.correctionExists)
			OperationsOnMatrixValues.incrementalAggregation(out.getValue(), correction.getValue(), 
				tagged.getBaseObject(), (AggregateOperator)instruction.getOperator());
		else
			OperationsOnMatrixValues.incrementalAggregation(out.getValue(), null, 
					tagged.getBaseObject(), (AggregateOperator)instruction.getOperator());	
	}
	*/
	//process one aggregate instruction
	private void processAggregateHelp(long row, long col, MatrixValue value, 
			AggregateInstruction instruction, boolean imbededCorrection) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		AggregateOperator aggOp=(AggregateOperator)instruction.getOperator();
		
		//there should be just one value in cache.
		IndexedMatrixValue out=cachedValues.getFirst(instruction.output);
	
		IndexedMatrixValue correction=null;
		if(aggOp.correctionExists)// && !imbededCorrection)
		{
			correction=correctionCache.getFirst(instruction.output);
		}
		
		if(out==null)
		{
			out=cachedValues.holdPlace(instruction.output, valueClass);
			out.getIndexes().setIndexes(row, col);
			//System.out.println("out: "+out);
			if(aggOp.correctionExists)// && !imbededCorrection)
			{
				if(correction==null)
					correction=correctionCache.holdPlace(instruction.output, valueClass);
				OperationsOnMatrixValues.startAggregation(out.getValue(), correction.getValue(), aggOp, 
						value.getNumRows(), value.getNumColumns(),
						value.isInSparseFormat(), imbededCorrection);
				
			}else
				OperationsOnMatrixValues.startAggregation(out.getValue(), null, aggOp, 
						value.getNumRows(), value.getNumColumns(),
						value.isInSparseFormat(), imbededCorrection);
			//System.out.println("after start: "+out);
		}
		//System.out.println("value to add: "+value);
		if(aggOp.correctionExists)// && !imbededCorrection)
		{
			//System.out.println("incremental aggregation maxindex/minindex");
			OperationsOnMatrixValues.incrementalAggregation(out.getValue(), correction.getValue(), 
					value, (AggregateOperator)instruction.getOperator(), imbededCorrection);
		}
		else
			OperationsOnMatrixValues.incrementalAggregation(out.getValue(), null, 
					value, (AggregateOperator)instruction.getOperator(), imbededCorrection);
		//System.out.println("after increment: "+out);
	}
	
	//process all the aggregate instructions for one group of values
	protected void processAggregateInstructions(MatrixIndexes indexes, Iterator<TaggedMatrixValue> values) 
		throws IOException
	{
		processAggregateInstructions(indexes, values, false);
	}
	
	//process all the aggregate instructions for one group of values
	protected void processAggregateInstructions(MatrixIndexes indexes, Iterator<TaggedMatrixValue> values, boolean imbededCorrection) 
	throws IOException
	{
		try
		{			
			while(values.hasNext())
			{
				TaggedMatrixValue value=values.next();
				byte input=value.getTag();
				ArrayList<AggregateInstruction> instructions=agg_instructions.get(input);
				
				//if there is no specified aggregate operation on an input, by default apply sum
				if(instructions==null)
				{
					defaultAggIns.input=input;
					defaultAggIns.output=input;	
					processAggregateHelp(indexes.getRowIndex(), indexes.getColumnIndex(), 
							value.getBaseObject(), defaultAggIns, imbededCorrection);
				}else //otherwise, perform the specified aggregate instructions
				{
					for(AggregateInstruction ins: instructions)
						processAggregateHelp(indexes.getRowIndex(), indexes.getColumnIndex(), 
								value.getBaseObject(), ins, imbededCorrection);
				}
			}
		}
		catch(Exception e)
		{
			throw new IOException(e);
		}
	}
	

	/**
	 * 
	 * @return
	 */
	protected boolean containsTernaryInstruction()
	{
		if( mixed_instructions != null )
			for(MRInstruction inst : mixed_instructions)
				if( inst instanceof TernaryInstruction )
					return true;
		return false;
	}
	
	protected boolean dimsKnownForTernaryInstructions() {
		if( mixed_instructions != null )
			for(MRInstruction inst : mixed_instructions)
				if( inst instanceof TernaryInstruction && !((TernaryInstruction)inst).knownOutputDims() )
					return false;
		return true;
	}

	/**
	 * 
	 * @param job
	 */
	protected void prepareMatrixCharacteristicsTernaryInstruction(JobConf job)
	{
		if( mixed_instructions != null )
			for(MRInstruction inst : mixed_instructions)
				if( inst instanceof TernaryInstruction )
				{
					TernaryInstruction tinst = (TernaryInstruction) inst;
					if( tinst.input1!=-1 )
						dimensions.put(tinst.input1, MRJobConfiguration.getMatrixCharacteristicsForInput(job, tinst.input1));					
					//extend as required, currently only ctableexpand needs blocksizes
				}
	}
}
