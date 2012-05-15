package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Vector;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MRInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.io.TaggedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.TaggedPartialBlock;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class ReduceBase extends MRBaseForCommonInstructions{
	
	protected static final Log LOG = LogFactory.getLog(ReduceBase.class);
	
	//aggregate instructions
	protected HashMap<Byte, Vector<AggregateInstruction>> 
	agg_instructions=new HashMap<Byte, Vector<AggregateInstruction>>();
	
	//default aggregate operation
	protected static final AggregateOperator DEFAULT_AGG_OP
	=new AggregateOperator(0, Plus.getPlusFnObject());

	//default aggregate instruction
	protected AggregateInstruction defaultAggIns=
		new AggregateInstruction(DEFAULT_AGG_OP, (byte)0, (byte)0, "DEFAULT_AGG_OP");
	
	//mixsure of instructions performed in reducer
	protected MRInstruction[] mixed_instructions= null;
	
	//the final result indexes that needed to be outputted
	protected byte[] resultIndexes=null;
	protected byte[] resultDimsUnknown=null;

	//output converters
	protected CollectMultipleConvertedOutputs collectFinalMultipleOutputs;
	
	//a counter to calculate the time spent in a reducer or a combiner
	protected static enum Counters {COMBINE_OR_REDUCE_TIME };

	//the counters to record how many nonZero cells have been produced for each output
	protected long[] resultsNonZeros=null;
	protected long[] resultsMaxRowDims=null;
	protected long[] resultsMaxColDims=null;
	
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

		
		//get the indexes of the final output matrices
		resultIndexes=MRJobConfiguration.getResultIndexes(job);
		resultDimsUnknown = MRJobConfiguration.getResultDimsUnknown(job);
		
		//initialize SystemML Counters (defined in MRJobConfiguration)
		resultsNonZeros=new long[resultIndexes.length];
		resultsMaxRowDims=new long[resultIndexes.length];
		resultsMaxColDims=new long[resultIndexes.length];
		Arrays.fill(resultsNonZeros,   0);
		Arrays.fill(resultsMaxRowDims, 0);
		Arrays.fill(resultsMaxColDims, 0);
		
		collectFinalMultipleOutputs = MRJobConfiguration.getMultipleConvertedOutputs(job);
		
		//parse aggregate operations
		AggregateInstruction[] agg_insts=null;
		try {
			agg_insts = MRJobConfiguration.getAggregateInstructions(job);
			//parse unary and binary operations
			mixed_instructions=MRJobConfiguration.getInstructionsInReducer(job);
			
		} catch (DMLUnsupportedOperationException e) {
			throw new RuntimeException(e);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}		
		
		//reorganize the aggregate instructions, so that they are all associatied with each input
		if(agg_insts!=null)
		{
			for(AggregateInstruction ins: agg_insts)
			{
				//associate instruction to its input
				Vector<AggregateInstruction> vec=agg_instructions.get(ins.input);
				if(vec==null)
				{
					vec=new Vector<AggregateInstruction>();
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
					vec=new Vector<AggregateInstruction>();
					agg_instructions.put(partialIns.input, vec);
				}
				vec.add(partialIns);
			}
		}
	}
	
	protected void collectOutput_N_Increase_Counter(MatrixIndexes indexes, MatrixValue value, 
			int i, Reporter reporter) throws IOException
	{
 		collectFinalMultipleOutputs.collectOutput(indexes, value, i, reporter);
		resultsNonZeros[i]+=value.getNonZeros();
		//TODO: remove redundant code
		//System.out.println(indexes+"\n"+value);
		//LOG.info("~~ output: "+indexes+"\n"+value);
		if ( resultDimsUnknown[i] == (byte) 1 ) {
			// compute dimensions for the resulting matrix
			
			// find the maximum row index and column index encountered in current output block/cell 
			long maxrow=0, maxcol=0;
		/*	try {
				maxrow = UtilFunctions.cellIndexCalculation( cachedValues.get(resultIndexes[i]).getIndexes().getRowIndex(), 
						cachedValues.get(resultIndexes[i]).getValue().getNumRows(), cachedValues.get(resultIndexes[i]).getValue().getMaxRow() );
				
				maxcol = UtilFunctions.cellIndexCalculation( cachedValues.get(resultIndexes[i]).getIndexes().getColumnIndex(), 
						cachedValues.get(resultIndexes[i]).getValue().getNumColumns(), cachedValues.get(resultIndexes[i]).getValue().getMaxColumn() );
			} catch(DMLRuntimeException e) {
				e.printStackTrace();
			}*/
			try {
				maxrow = value.getMaxRow();
				maxcol = value.getMaxColumn();
				//System.out.println("maxrow = " + maxrow + ", maxcol = " + maxcol + ", val = " + value.getValue((int)indexes.getRowIndex(), (int)indexes.getColumnIndex()));
			} catch (DMLRuntimeException e) {
				throw new IOException(e);
			}
			
			if ( maxrow > resultsMaxRowDims[i] )
				resultsMaxRowDims[i] = maxrow;
				
			if ( maxcol > resultsMaxColDims[i] )
				resultsMaxColDims[i] = maxcol;
		}else if(resultDimsUnknown[i] == (byte) 2)
		{
			if ( indexes.getRowIndex() > resultsMaxRowDims[i] )
				resultsMaxRowDims[i] = indexes.getRowIndex();
				
			if ( indexes.getColumnIndex() > resultsMaxColDims[i] )
				resultsMaxColDims[i] = indexes.getColumnIndex();
			//System.out.println("i = " + i + ", maxrow = " + resultsMaxRowDims[i] + ", maxcol = " + resultsMaxColDims[i] + ", val = " + value.getValue((int)indexes.getRowIndex(), (int)indexes.getColumnIndex()));
		}
	}
	
	protected Vector<Integer> getOutputIndexes(byte outputTag)
	{
		Vector<Integer> ret=new Vector<Integer>();
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
			
			for(int i=0; i<resultIndexes.length; i++) {
				cachedReporter.incrCounter(MRJobConfiguration.NUM_NONZERO_CELLS, Integer.toString(i), resultsNonZeros[i]);
				
				if ( resultDimsUnknown!=null && resultDimsUnknown[i] != (byte) 0 ) {
					// Each counter is of the form: (group, name)
					// where group = max_rowdim_resultindex; name = taskid
					//System.out.println("--> before i="+i+", row = " + cachedReporter.getCounter("max_rowdim_"+i, ""+taskid).getCounter() + ", col = " + cachedReporter.getCounter("max_coldim_"+i, ""+taskid).getCounter());
					cachedReporter.getCounter("max_rowdim_"+i, ""+taskid).increment(resultsMaxRowDims[i]);
					cachedReporter.getCounter("max_coldim_"+i, ""+taskid).increment(resultsMaxColDims[i]);
					//System.out.println("--> after i="+i+", row = " + cachedReporter.getCounter("max_rowdim_"+i, ""+taskid).getCounter() + ", col = " + cachedReporter.getCounter("max_coldim_"+i, ""+taskid).getCounter());
				}
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
	
	protected void processReblockInReducer(MatrixIndexes indexes, Iterator<TaggedPartialBlock> values, 
			HashMap<Byte, MatrixCharacteristics> dimensions)
	{
		while(values.hasNext())
		{
			TaggedPartialBlock partial=values.next();
		//	System.out.println("in Reducer: "+indexes+": "+partial);
			Byte tag=partial.getTag();
			
			//there only one block in the cache for this output
			IndexedMatrixValue block=cachedValues.getFirst(tag);
			
			if(block==null)
			{
				block=cachedValues.holdPlace(tag, valueClass);
				int brlen=dimensions.get(tag).numRowsPerBlock;
				int bclen=dimensions.get(tag).numColumnsPerBlock;
				int realBrlen=(int)Math.min((long)brlen, dimensions.get(tag).numRows-(indexes.getRowIndex()-1)*brlen);
				int realBclen=(int)Math.min((long)bclen, dimensions.get(tag).numColumns-(indexes.getColumnIndex()-1)*bclen);
				block.getValue().reset(realBrlen, realBclen);
				block.getIndexes().setIndexes(indexes);
			}
			int row=partial.getBaseObject().getRowIndex();
			int column=partial.getBaseObject().getColumnIndex();
			if(row>=0 && column >=0)
				block.getValue().setValue(row, column, partial.getBaseObject().getValue());
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
		}
		
		if(aggOp.correctionExists)// && !imbededCorrection)
			OperationsOnMatrixValues.incrementalAggregation(out.getValue(), correction.getValue(), 
					value, (AggregateOperator)instruction.getOperator(), imbededCorrection);
		else
			OperationsOnMatrixValues.incrementalAggregation(out.getValue(), null, 
					value, (AggregateOperator)instruction.getOperator(), imbededCorrection);	
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
		while(values.hasNext())
		{
			TaggedMatrixValue value=values.next();
			byte input=value.getTag();
			Vector<AggregateInstruction> instructions=agg_instructions.get(input);
			
		//	System.out.println("value to aggregate: "+value);
			
			//if there is no specified aggregate operation on an input, by default apply sum
			try{
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
			}catch(Exception e)
			{
				throw new IOException(e);
			}
		//	System.out.println("current cachedValues: \n"+cachedValues);
		}
	}
	
}
