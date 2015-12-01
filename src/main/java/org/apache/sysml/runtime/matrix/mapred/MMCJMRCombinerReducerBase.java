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


package org.apache.sysml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.instructions.mr.AggregateBinaryInstruction;
import org.apache.sysml.runtime.instructions.mr.AggregateInstruction;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.TaggedFirstSecondIndexes;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;


public class MMCJMRCombinerReducerBase extends ReduceBase
{
		
	//aggregate binary instruction for the mmcj
	protected AggregateBinaryInstruction aggBinInstruction=null;
	
	//temporary variable to hold the aggregate result
	protected MatrixValue buffer=null;
	
	//the tags to be output for the left and right matrice for the mmcj
	protected byte tagForLeft=0;
	protected byte tagForRight=1;
	protected MatrixCharacteristics dim1;
	protected MatrixCharacteristics dim2;
//	protected int elementSize=8;

	public void configure(JobConf job)
	{
		super.configure(job);
		
		AggregateBinaryInstruction[] ins;
		try {
			ins = MRJobConfiguration.getAggregateBinaryInstructions(job);
		} catch (DMLUnsupportedOperationException e) {
			throw new RuntimeException(e);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
		if(ins.length!=1)
			throw new RuntimeException("MMCJ only perform one aggregate binary instruction");
		aggBinInstruction=ins[0];
		
		//decide which matrix need to be cached for cross product
		dim1=MRJobConfiguration.getMatrixCharactristicsForBinAgg(job, aggBinInstruction.input1);
		dim2=MRJobConfiguration.getMatrixCharactristicsForBinAgg(job, aggBinInstruction.input2);
		if(dim1.getRows()>dim2.getCols())
		{
			tagForLeft=1;
			tagForRight=0;
		}

		//allocate space for the temporary variable
		try {
			buffer=valueClass.newInstance();
		} catch (Exception e) {
			throw new RuntimeException(e);
		} 
		
//		if(valueClass.equals(MatrixCell.class))
//			elementSize=90;
	}
	
	protected MatrixValue performAggregateInstructions(TaggedFirstSecondIndexes indexes, Iterator<MatrixValue> values) 
	throws IOException
	{
		//manipulation on the tags first
		byte realTag=indexes.getTag();
		byte representTag;
		if(realTag==tagForLeft)
			representTag=aggBinInstruction.input1;
		else
			representTag=aggBinInstruction.input2;
		
		ArrayList<AggregateInstruction> instructions=agg_instructions.get(representTag);
		AggregateInstruction ins;
		if(instructions==null)
		{
			defaultAggIns.input=realTag;
			defaultAggIns.output=realTag;
			ins=defaultAggIns;
		}else
		{
			if(instructions.size()>1)
				throw new IOException("only one aggregate operation on input "
						+indexes.getTag()+" is allowed in BlockMMCJMR");
			ins=instructions.get(0);
			if(ins.input!=ins.output)
				throw new IOException("input index and output index have to be " +
						"the same for aggregate instructions in BlockMMCJMR");
		}
		
		//performa aggregation before doing mmcj
		//TODO: customize the code, since aggregation for matrix multiplcation can only be sum
		boolean needStartAgg=true;
		try {
			while(values.hasNext())
			{
				MatrixValue value=values.next();
				if(needStartAgg)
				{
					buffer.reset(value.getNumRows(), value.getNumColumns(), value.isInSparseFormat());
					needStartAgg=false;
				//	LOG.info("initialize buffer: sparse="+buffer.isInSparseFormat()+", nonZero="+buffer.getNonZeros());
				}
				buffer.binaryOperationsInPlace(((AggregateOperator)ins.getOperator()).increOp, value);
			//	LOG.info("increment buffer: sparse="+buffer.isInSparseFormat()+", nonZero="+buffer.getNonZeros());
			}
		} catch (Exception e) {
			throw new IOException(e);
		}
		
		if(needStartAgg)
			return null;
		else
			return buffer;
	}
}
