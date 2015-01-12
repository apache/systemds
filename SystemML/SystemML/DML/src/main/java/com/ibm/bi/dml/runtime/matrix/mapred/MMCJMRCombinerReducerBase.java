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

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.mr.AggregateBinaryInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.AggregateInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;


public class MMCJMRCombinerReducerBase extends ReduceBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
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
		if(dim1.numRows>dim2.numColumns)
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
