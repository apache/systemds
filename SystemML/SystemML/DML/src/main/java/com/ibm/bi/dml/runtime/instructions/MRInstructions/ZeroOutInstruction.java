/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import java.util.ArrayList;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RangeBasedReIndexInstruction.IndexRange;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReIndexOperator;
import com.ibm.bi.dml.runtime.matrix.operators.ZeroOutOperator;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
/*
 * ZeroOut with complementary=false is to zero out a subregion inside a matrix
 * ZeroOut with complementary=true is to select a subregion inside a matrix (zero out regions outside the selected range)
 */
public class ZeroOutInstruction extends UnaryMRInstructionBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	public IndexRange indexRange=null;
	private IndexRange tempRange=new IndexRange(-1, -1, -1, -1);
	public boolean complementary=false;
	
	public ZeroOutInstruction(Operator op, byte in, byte out, IndexRange rng, String istr) {
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.ZeroOut;
		instString = istr;
		indexRange=rng;
	}
	
	private static long parseStartBoundary(String str)
	{
		if(!str.isEmpty())
			return Long.parseLong(str);
		else
			return 1;
	}
	
	private static long parseEndBoundary(String str)
	{
		if(!str.isEmpty())
			return Long.parseLong(str);
		else
			return Long.MAX_VALUE;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 6 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		String opcode = parts[0];
		if(!opcode.equalsIgnoreCase("zeroOut"))
			throw new DMLRuntimeException("Unknown opcode while parsing a zeroout: " + str);
		byte in = Byte.parseByte(parts[1]);

		//IndexRange rng=new IndexRange(Long.parseLong(parts[2]), Long.parseLong(parts[3]), Long.parseLong(parts[4]), Long.parseLong(parts[5]));
		IndexRange rng=new IndexRange(UtilFunctions.parseToLong(parts[2]), 
				UtilFunctions.parseToLong(parts[3]), 
				UtilFunctions.parseToLong(parts[4]), 
				UtilFunctions.parseToLong(parts[5]));
		byte out = Byte.parseByte(parts[6]);
		return new ZeroOutInstruction(new ZeroOutOperator(), in, out, rng, str);
	}

	private IndexRange getSelectedRange(IndexedMatrixValue in, int blockRowFactor, int blockColFactor) {
		
		long topBlockRowIndex=UtilFunctions.blockIndexCalculation(indexRange.rowStart, blockRowFactor);
		int topRowInTopBlock=UtilFunctions.cellInBlockCalculation(indexRange.rowStart, blockRowFactor);
		long bottomBlockRowIndex=UtilFunctions.blockIndexCalculation(indexRange.rowEnd, blockRowFactor);
		int bottomRowInBottomBlock=UtilFunctions.cellInBlockCalculation(indexRange.rowEnd, blockRowFactor);
		
		long leftBlockColIndex=UtilFunctions.blockIndexCalculation(indexRange.colStart, blockColFactor);
		int leftColInLeftBlock=UtilFunctions.cellInBlockCalculation(indexRange.colStart, blockColFactor);
		long rightBlockColIndex=UtilFunctions.blockIndexCalculation(indexRange.colEnd, blockColFactor);
		int rightColInRightBlock=UtilFunctions.cellInBlockCalculation(indexRange.colEnd, blockColFactor);
		
		//no overlap
		if(in.getIndexes().getRowIndex()<topBlockRowIndex || in.getIndexes().getRowIndex()>bottomBlockRowIndex
		   || in.getIndexes().getColumnIndex()<leftBlockColIndex || in.getIndexes().getColumnIndex()>rightBlockColIndex)
		{
			tempRange.set(-1,-1,-1,-1);
			return tempRange;
		}
		
		//get the index range inside the block
		tempRange.set(0, in.getValue().getNumRows()-1, 0, in.getValue().getNumColumns()-1);
		if(topBlockRowIndex==in.getIndexes().getRowIndex())
			tempRange.rowStart=topRowInTopBlock;
		if(bottomBlockRowIndex==in.getIndexes().getRowIndex())
			tempRange.rowEnd=bottomRowInBottomBlock;
		if(leftBlockColIndex==in.getIndexes().getColumnIndex())
			tempRange.colStart=leftColInLeftBlock;
		if(rightBlockColIndex==in.getIndexes().getColumnIndex())
			tempRange.colEnd=rightColInRightBlock;
		
		return tempRange;
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(input);
		
		if( blkList != null )
			for(IndexedMatrixValue in : blkList)
			{
				if(in==null)
					continue;
			
				tempRange=getSelectedRange(in, blockRowFactor, blockColFactor);
				if(tempRange.rowStart==-1 && complementary)//just selection operation
					return;
				
				if(tempRange.rowStart==-1 && !complementary)//if no overlap, directly write them out
				{
					cachedValues.add(output, in);
					//System.out.println("just write down: "+in);
					return;
				}
				
				//allocate space for the output value
				IndexedMatrixValue out;
				if(input==output)
					out=tempValue;
				else
					out=cachedValues.holdPlace(output, valueClass);
				
				//process instruction
				
				OperationsOnMatrixValues.performZeroOut(in.getIndexes(), in.getValue(), 
						out.getIndexes(), out.getValue(), tempRange, complementary);
				
				//put the output value in the cache
				if(out==tempValue)
					cachedValues.add(output, out);
			}
		
	}
	
	public static void main(String[] args) throws Exception {
		
		byte input=1;
		byte output=2;
		ZeroOutInstruction ins=new ZeroOutInstruction(new ReIndexOperator(), input, output, new IndexRange(3, 8, 3, 18), "zeroOut");
		int blockRowFactor=10;
		int blockColFactor=10;
		
		MatrixBlockDSM m=MatrixBlockDSM.getRandomSparseMatrix(blockRowFactor, blockColFactor, 1, 1);
		//m.examSparsity();
		CachedValueMap cachedValues=new CachedValueMap();
		cachedValues.set(input, new MatrixIndexes(1, 1), m);
		
		IndexedMatrixValue tempValue=new IndexedMatrixValue(MatrixBlockDSM.class);
		IndexedMatrixValue zeroInput=new IndexedMatrixValue(MatrixBlockDSM.class);
		
		ins.processInstruction(MatrixBlockDSM.class, cachedValues, tempValue, zeroInput, blockRowFactor, blockColFactor);
		System.out.println(cachedValues.get(output));
	}
}
