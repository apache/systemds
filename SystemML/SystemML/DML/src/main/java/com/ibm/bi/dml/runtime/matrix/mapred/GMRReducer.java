/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.TertiaryInstruction;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixPackedCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.TaggedMatrixValue;


public class GMRReducer extends ReduceBase
implements Reducer<MatrixIndexes, TaggedMatrixValue, MatrixIndexes, MatrixValue>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private MatrixValue realOutValue;
	private GMRCtableBuffer _buff;
	
	public void reduce(MatrixIndexes indexes, Iterator<TaggedMatrixValue> values,
			OutputCollector<MatrixIndexes, MatrixValue> out, Reporter reporter) 
		throws IOException 
	{
		long start=System.currentTimeMillis();
		commonSetup(reporter);
		
		cachedValues.reset();
	
		//perform aggregate operations first
		processAggregateInstructions(indexes, values);
		
		//perform mixed operations
		try {
			processReducerInstructionsInGMR();
		} catch (Exception e) {
			throw new IOException(e);
		} 
		
		//output the final result matrices
		outputResultsFromCachedValuesForGMR(reporter);

		reporter.incrCounter(Counters.COMBINE_OR_REDUCE_TIME, System.currentTimeMillis()-start);
	}
	
	//process mixture of instructions
	protected void processReducerInstructionsInGMR() throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		if(mixed_instructions==null)
			return;
		for(MRInstruction ins: mixed_instructions)
		{
			if(ins instanceof TertiaryInstruction)
			{  
				((TertiaryInstruction) ins).processInstruction(valueClass, cachedValues, zeroInput, _buff.getBuffer());
				if( _buff.getBufferSize() > GMRCtableBuffer.MAX_BUFFER_SIZE )
					_buff.flushBuffer(cachedReporter); //prevent oom for large/many ctables
			}
			else
				processOneInstruction(ins, valueClass, cachedValues, tempValue, zeroInput);
		}
	}
	
	protected void outputResultsFromCachedValuesForGMR(Reporter reporter) throws IOException
	{
		for(int i=0; i<resultIndexes.length; i++)
		{
			byte output=resultIndexes[i];
			IndexedMatrixValue outValue=cachedValues.getFirst(output);
			if(outValue==null)
				continue;
			if(valueClass.equals(MatrixPackedCell.class))
			{
				realOutValue.copy(outValue.getValue());
				collectOutput_N_Increase_Counter(outValue.getIndexes(), 
						realOutValue, i, reporter);
			}
			else
				collectOutput_N_Increase_Counter(outValue.getIndexes(), 
					outValue.getValue(), i, reporter);
	//		LOG.info("output: "+outValue.getIndexes()+" -- "+outValue.getValue()+" ~~ tag: "+output);
	//		System.out.println("Reducer output: "+outValue.getIndexes()+" -- "+outValue.getValue()+" ~~ tag: "+output);
		}
	}
	
	public void close()throws IOException
	{
		//flush ctable buffer 
		_buff.flushBuffer(cachedReporter);
		
			
		super.close();		
	}
	
	public void configure(JobConf job)
	{
		super.configure(job);
		
		_buff = new GMRCtableBuffer(collectFinalMultipleOutputs);
		_buff.setMetadataReferences(resultIndexes, resultsNonZeros, resultDimsUnknown, resultsMaxRowDims, resultsMaxColDims);
		
		try {
			realOutValue=valueClass.newInstance();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		//this is to make sure that aggregation works for GMR
		if(valueClass.equals(MatrixCell.class))
			valueClass=MatrixPackedCell.class;
	}
}
