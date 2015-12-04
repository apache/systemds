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


package org.apache.sysml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import org.apache.sysml.runtime.instructions.mr.AppendRInstruction;
import org.apache.sysml.runtime.instructions.mr.MRInstruction;
import org.apache.sysml.runtime.instructions.mr.TernaryInstruction;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixPackedCell;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.TaggedMatrixValue;


public class GMRReducer extends ReduceBase
implements Reducer<MatrixIndexes, TaggedMatrixValue, MatrixIndexes, MatrixValue>
{
		
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
		processReducerInstructionsInGMR(indexes);
		
		//output the final result matrices
		outputResultsFromCachedValuesForGMR(reporter);

		reporter.incrCounter(Counters.COMBINE_OR_REDUCE_TIME, System.currentTimeMillis()-start);
	}
	
	/**
	 * 
	 * @throws IOException
	 */
	protected void processReducerInstructionsInGMR(MatrixIndexes indexes) 
		throws IOException 
	{
		if(mixed_instructions==null)
			return;
		
		try 
		{		
			for(MRInstruction ins: mixed_instructions)
			{
				if(ins instanceof TernaryInstruction)
				{  
					MatrixCharacteristics dim = dimensions.get(((TernaryInstruction) ins).input1);
					((TernaryInstruction) ins).processInstruction(valueClass, cachedValues, zeroInput, _buff.getMapBuffer(), _buff.getBlockBuffer(), dim.getRowsPerBlock(), dim.getColsPerBlock());
					if( _buff.getBufferSize() > GMRCtableBuffer.MAX_BUFFER_SIZE )
						_buff.flushBuffer(cachedReporter); //prevent oom for large/many ctables
				}
				else if(ins instanceof AppendRInstruction) {
					MatrixCharacteristics dims1 = dimensions.get(((AppendRInstruction) ins).input1);
					MatrixCharacteristics dims2 = dimensions.get(((AppendRInstruction) ins).input2);
					long nbi1 = (long) Math.ceil((double)dims1.getRows()/dims1.getRowsPerBlock());
					long nbi2 = (long) Math.ceil((double)dims2.getRows()/dims2.getRowsPerBlock());
					long nbj1 = (long) Math.ceil((double)dims1.getCols()/dims1.getColsPerBlock());
					long nbj2 = (long) Math.ceil((double)dims2.getCols()/dims2.getColsPerBlock());
					
					// Execute the instruction only if current indexes fall within the range of input dimensions
					if((nbi1 < indexes.getRowIndex() && nbi2 < indexes.getRowIndex()) || (nbj1 < indexes.getColumnIndex() && nbj2 < indexes.getColumnIndex()))
						continue;
					else
						processOneInstruction(ins, valueClass, cachedValues, tempValue, zeroInput);
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
		if( containsTernaryInstruction() ){
			_buff = new GMRCtableBuffer(collectFinalMultipleOutputs, dimsKnownForTernaryInstructions());
			_buff.setMetadataReferences(resultIndexes, resultsNonZeros, resultDimsUnknown, resultsMaxRowDims, resultsMaxColDims);
			prepareMatrixCharacteristicsTernaryInstruction(job); //put matrix characteristics in dimensions map
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
		if( containsTernaryInstruction() )
			_buff.flushBuffer(cachedReporter);
					
		super.close();		
	}
	
	
}
