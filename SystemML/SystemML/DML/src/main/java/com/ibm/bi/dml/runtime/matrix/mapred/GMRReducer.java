/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.instructions.mr.MRInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.TertiaryInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixPackedCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.TaggedMatrixValue;


public class GMRReducer extends ReduceBase
implements Reducer<MatrixIndexes, TaggedMatrixValue, MatrixIndexes, MatrixValue>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private MatrixValue realOutValue;
	private GMRCtableBuffer _buff;
	
	@Override
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
		processReducerInstructionsInGMR();
		
		//output the final result matrices
		outputResultsFromCachedValuesForGMR(reporter);

		reporter.incrCounter(Counters.COMBINE_OR_REDUCE_TIME, System.currentTimeMillis()-start);
	}
	
	/**
	 * 
	 * @throws IOException
	 */
	protected void processReducerInstructionsInGMR() 
		throws IOException 
	{
		if(mixed_instructions==null)
			return;
		
		try 
		{		
			for(MRInstruction ins: mixed_instructions)
			{
				if(ins instanceof TertiaryInstruction)
				{  
					MatrixCharacteristics dim = dimensions.get(((TertiaryInstruction) ins).input1);
					((TertiaryInstruction) ins).processInstruction(valueClass, cachedValues, zeroInput, _buff.getBuffer(), _buff.getBlockBuffer(), dim.getRowsPerBlock(), dim.getColsPerBlock());
					if( _buff.getBufferSize() > GMRCtableBuffer.MAX_BUFFER_SIZE )
						_buff.flushBuffer(cachedReporter); //prevent oom for large/many ctables
				}
				else
					processOneInstruction(ins, valueClass, cachedValues, tempValue, zeroInput);
			}
		} 
		catch (Exception e) {
			throw new IOException(e);
		}
	}
	
	/**
	 * 
	 * @param reporter
	 * @throws IOException
	 */
	protected void outputResultsFromCachedValuesForGMR(Reporter reporter) throws IOException
	{
		for(int i=0; i<resultIndexes.length; i++)
		{
			byte output=resultIndexes[i];
			
			ArrayList<IndexedMatrixValue> outValueList = cachedValues.get(output);
			if( outValueList == null ) 
				continue;
		
			for(IndexedMatrixValue outValue : outValueList) //for all blocks of given index
			{
				if(valueClass.equals(MatrixPackedCell.class)) {
					realOutValue.copy(outValue.getValue());
					collectOutput_N_Increase_Counter(outValue.getIndexes(), realOutValue, i, reporter);
				}
				else
					collectOutput_N_Increase_Counter(outValue.getIndexes(), outValue.getValue(), i, reporter);		
			}
		}
	}
	
	@Override
	public void configure(JobConf job)
	{
		super.configure(job);
		
		//init ctable buffer (if required, after super init)
		if( containsTertiaryInstruction() ){
			_buff = new GMRCtableBuffer(collectFinalMultipleOutputs, dimsKnownForTertiaryInstructions());
			_buff.setMetadataReferences(resultIndexes, resultsNonZeros, resultDimsUnknown, resultsMaxRowDims, resultsMaxColDims);
			prepareMatrixCharacteristicsTertiaryInstruction(job); //put matrix characteristics in dimensions map
		}
		
		try {
			realOutValue=valueClass.newInstance();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		//this is to make sure that aggregation works for GMR
		if(valueClass.equals(MatrixCell.class))
			valueClass=MatrixPackedCell.class;
	}
	
	@Override
	public void close()throws IOException
	{
		//flush ctable buffer (if required)
		if( containsTertiaryInstruction() )
			_buff.flushBuffer(cachedReporter);
					
		super.close();		
	}
	
	
}
