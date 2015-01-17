/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.mr;

import java.util.ArrayList;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReIndexOperator;
import com.ibm.bi.dml.runtime.util.UtilFunctions;


public class RangeBasedReIndexInstruction extends UnaryMRInstructionBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected boolean forLeftIndexing=false;
	protected long leftMatrixNRows=0;
	protected long leftMatrixNCols=0;
	
	private IndexRange indexRange=null;
	private IndexRange tempRange=new IndexRange(-1, -1, -1, -1);
	
	
	//start and end are all inclusive
	public static class IndexRange
	{
		public long rowStart=0;
		public long rowEnd=0;
		public long colStart=0;
		public long colEnd=0;
		
		public IndexRange(long rs, long re, long cs, long ce)
		{
			set(rs, re, cs, ce);
		}
		public void set(long rs, long re, long cs, long ce)
		{
			//if(re<rs || ce<cs)
			//	throw new RuntimeException("This is invalid index range: ["+rs+":"+re+", "+cs+":"+ce+"]!");
			rowStart=rs;
			rowEnd=re;
			colStart=cs;
			colEnd=ce;
		}
	}
	
	public RangeBasedReIndexInstruction(Operator op, byte in, byte out, IndexRange rng, String istr) {
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.RangeReIndex;
		instString = istr;
		indexRange=rng;
	}
	
	public RangeBasedReIndexInstruction(Operator op, byte in, byte out, IndexRange rng, boolean forleft, long leftNRows, long leftNCols, String istr) {
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.RangeReIndex;
		instString = istr;
		indexRange=rng;
		this.forLeftIndexing=forleft;
		this.leftMatrixNRows=leftNRows;
		this.leftMatrixNCols=leftNCols;
	}
	
	public IndexRange getIndexRange() {
		return indexRange;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 8 );
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		String opcode = parts[0];
		boolean forLeft=false;
		if(opcode.equalsIgnoreCase("rangeReIndexForLeft"))
			forLeft=true;
		else if(!opcode.equalsIgnoreCase("rangeReIndex"))
			throw new DMLRuntimeException("Unknown opcode while parsing a Select: " + str);
		byte in = Byte.parseByte(parts[1]); 
		IndexRange rng=new IndexRange(UtilFunctions.parseToLong(parts[2]), 
									  UtilFunctions.parseToLong(parts[3]), 
									  UtilFunctions.parseToLong(parts[4]),
									  UtilFunctions.parseToLong(parts[5]));
		byte out = Byte.parseByte(parts[6]);
		long leftIndexingNrow=Long.parseLong(parts[7]);
		long leftIndexingNcol=Long.parseLong(parts[8]);
		
		//recalculate the index range for left indexing
		if(forLeft)
		{
			long a=rng.rowStart;
			long b=rng.colStart;
			rng.rowStart=2-a;
			rng.colStart=2-b;
			//don't need to extend to the whole left matrix dimension
			rng.rowEnd=leftIndexingNrow-a+1;
			rng.colEnd=leftIndexingNcol-b+1;
			return new RangeBasedReIndexInstruction(new ReIndexOperator(), in, out, rng, forLeft, leftIndexingNrow, leftIndexingNcol, str);
		}else
			return new RangeBasedReIndexInstruction(new ReIndexOperator(), in, out, rng, str);
	}
	
	
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		
		if(input==output)
			throw new DMLRuntimeException("input cannot be the same for output for "+this.instString);
		
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(input);
		if( blkList != null )
			for(IndexedMatrixValue in : blkList)
			{
				if(in==null)
					continue;
	
				long cellIndexTopRow=UtilFunctions.cellIndexCalculation(in.getIndexes().getRowIndex(), blockRowFactor, 0);
				long cellIndexBottomRow=UtilFunctions.cellIndexCalculation(in.getIndexes().getRowIndex(), blockRowFactor, in.getValue().getNumRows()-1);
				long cellIndexLeftCol=UtilFunctions.cellIndexCalculation(in.getIndexes().getColumnIndex(), blockColFactor, 0);
				long cellIndexRightCol=UtilFunctions.cellIndexCalculation(in.getIndexes().getColumnIndex(), blockColFactor, in.getValue().getNumColumns()-1);
				
				long cellIndexOverlapTop=Math.max(cellIndexTopRow, indexRange.rowStart);
				long cellIndexOverlapBottom=Math.min(cellIndexBottomRow, indexRange.rowEnd);
				long cellIndexOverlapLeft=Math.max(cellIndexLeftCol, indexRange.colStart);
				long cellIndexOverlapRight=Math.min(cellIndexRightCol, indexRange.colEnd);
				
				if(cellIndexOverlapTop>cellIndexOverlapBottom || cellIndexOverlapLeft>cellIndexOverlapRight)
					return;
				
				tempRange.set(UtilFunctions.cellInBlockCalculation(cellIndexOverlapTop, blockRowFactor), 
						UtilFunctions.cellInBlockCalculation(cellIndexOverlapBottom, blockRowFactor), 
						UtilFunctions.cellInBlockCalculation(cellIndexOverlapLeft, blockColFactor), 
						UtilFunctions.cellInBlockCalculation(cellIndexOverlapRight, blockColFactor));
				
				int rowCut=UtilFunctions.cellInBlockCalculation(indexRange.rowStart, blockRowFactor);
				int colCut=UtilFunctions.cellInBlockCalculation(indexRange.colStart, blockColFactor);
				
				int rowsInLastBlock=(int)((indexRange.rowEnd-indexRange.rowStart+1)%blockRowFactor);
				if(rowsInLastBlock==0) rowsInLastBlock=blockRowFactor;
				int colsInLastBlock=(int)((indexRange.colEnd-indexRange.colStart+1)%blockColFactor);
				if(colsInLastBlock==0) colsInLastBlock=blockColFactor;
				
				long resultBlockIndexTop=UtilFunctions.blockIndexCalculation(cellIndexOverlapTop-indexRange.rowStart+1, blockRowFactor);
				long resultBlockIndexBottom=UtilFunctions.blockIndexCalculation(cellIndexOverlapBottom-indexRange.rowStart+1, blockRowFactor);
				long resultBlockIndexLeft=UtilFunctions.blockIndexCalculation(cellIndexOverlapLeft-indexRange.colStart+1, blockColFactor);
				long resultBlockIndexRight=UtilFunctions.blockIndexCalculation(cellIndexOverlapRight-indexRange.colStart+1, blockColFactor);
				
				int boundaryRlen=blockRowFactor, boundaryClen=blockColFactor;
				long finalBlockIndexBottom=UtilFunctions.blockIndexCalculation(indexRange.rowEnd-indexRange.rowStart+1, blockRowFactor);
				long finalBlockIndexRight=UtilFunctions.blockIndexCalculation(indexRange.colEnd-indexRange.colStart+1, blockColFactor);
				if(resultBlockIndexBottom==finalBlockIndexBottom)
					boundaryRlen=rowsInLastBlock;
				if(resultBlockIndexRight==finalBlockIndexRight)
					boundaryClen=colsInLastBlock;
					
				//allocate space for the output value
				ArrayList<IndexedMatrixValue> outlist=new ArrayList<IndexedMatrixValue>(4);
				for(long r=resultBlockIndexTop; r<=resultBlockIndexBottom; r++)
					for(long c=resultBlockIndexLeft; c<=resultBlockIndexRight; c++)
					{
						IndexedMatrixValue out=cachedValues.holdPlace(output, valueClass);
						out.getIndexes().setIndexes(r, c);
						outlist.add(out);
					}
				
				//process instruction
				OperationsOnMatrixValues.performSlice(in.getIndexes(), in.getValue(), outlist, tempRange, rowCut, colCut, blockRowFactor, blockColFactor, boundaryRlen, boundaryClen);
			}
	}
}
