/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.mr.AggregateBinaryInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.TaggedFirstSecondIndexes;


public class MMCJMRMapper extends MapperBase 
implements Mapper<Writable, Writable, Writable, Writable>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//the aggregate binary instruction for this mmcj job
	private AggregateBinaryInstruction aggBinInstruction;
	
	//tempory variable
	private TaggedFirstSecondIndexes taggedIndexes=new TaggedFirstSecondIndexes();
	
	//the tags to be output for the left and right matrice for the mmcj
	private byte tagForLeft=0;
	private byte tagForRight=1;
	
	@Override
	public void map(Writable rawKey, Writable rawValue,
			OutputCollector<Writable, Writable> out,
			Reporter reporter) throws IOException {
		
		commonMap(rawKey, rawValue, out, reporter);
	}

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
		MatrixCharacteristics dim1=MRJobConfiguration.getMatrixCharactristicsForBinAgg(job, aggBinInstruction.input1);
		MatrixCharacteristics dim2=MRJobConfiguration.getMatrixCharactristicsForBinAgg(job, aggBinInstruction.input2);
		if(dim1.getRows()>dim2.getCols())
		{
			tagForLeft=1;
			tagForRight=0;
		}
	}

	@Override
	protected void specialOperationsForActualMap(int index,
			OutputCollector<Writable, Writable> out, Reporter reporter)
			throws IOException {
		//apply all instructions
		processMapperInstructionsForMatrix(index);
		
		//process the mapper part of MMCJ
		processMMCJInMapperAndOutput(aggBinInstruction, tagForLeft, tagForRight, 
				taggedIndexes, out);
	}
	
	protected void processMMCJInMapperAndOutput(AggregateBinaryInstruction aggBinInstruction, 
			byte tagForLeft, byte tagForRight, TaggedFirstSecondIndexes taggedIndexes,
			OutputCollector<Writable, Writable> out) throws IOException
	{		
		//output the key value pair for the left matrix
		ArrayList<IndexedMatrixValue> blkList1 = cachedValues.get(aggBinInstruction.input1);
		if( blkList1 != null )
			for(IndexedMatrixValue result:blkList1)
				if(result!=null)
				{
					taggedIndexes.setTag(tagForLeft);
					taggedIndexes.setIndexes(result.getIndexes().getColumnIndex(), 
							result.getIndexes().getRowIndex());
					
					if( !((MatrixBlock)result.getValue()).isEmptyBlock() )
						out.collect(taggedIndexes, result.getValue());
					//System.out.println("In Mapper: output "+taggedIndexes+" "+ result.getValue().getNumRows()+"x"+result.getValue().getNumColumns());
				}
		
		//output the key value pair for the right matrix
		//Note: due to cached list reuse after first flush
		ArrayList<IndexedMatrixValue> blkList2 = cachedValues.get(aggBinInstruction.input2);
		if( blkList2 != null )
			for(IndexedMatrixValue result:blkList2)
				if(result!=null)
				{
					taggedIndexes.setTag(tagForRight);
					taggedIndexes.setIndexes(result.getIndexes().getRowIndex(), 
							result.getIndexes().getColumnIndex());
					
					if( !((MatrixBlock)result.getValue()).isEmptyBlock() ) 
						out.collect(taggedIndexes, result.getValue());
					//System.out.println("In Mapper: output "+taggedIndexes+" "+ result.getValue().getNumRows()+"x"+result.getValue().getNumColumns());
				}
	}
}
