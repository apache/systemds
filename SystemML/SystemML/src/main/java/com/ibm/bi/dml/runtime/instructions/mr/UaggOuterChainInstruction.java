/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.mr;

import java.util.ArrayList;
import java.util.Arrays;

import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.KahanPlus;
import com.ibm.bi.dml.runtime.functionobjects.LessThan;
import com.ibm.bi.dml.runtime.functionobjects.ReduceAll;
import com.ibm.bi.dml.runtime.functionobjects.ReduceCol;
import com.ibm.bi.dml.runtime.functionobjects.ReduceRow;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.DistributedCacheInput;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.MRBaseForCommonInstructions;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;


/**
 * 
 */
public class UaggOuterChainInstruction extends BinaryInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//operators
	private AggregateUnaryOperator _uaggOp = null;
	private AggregateOperator _aggOp = null;
	private BinaryOperator _bOp = null;
	
	//reused intermediates  
	private MatrixValue _tmpVal1 = null;
	private MatrixValue _tmpVal2 = null;
	
	private double[] _bv = null;
	
	/**
	 * 
	 * @param bop
	 * @param uaggop
	 * @param in1
	 * @param out
	 * @param istr
	 */
	public UaggOuterChainInstruction(BinaryOperator bop, AggregateUnaryOperator uaggop, AggregateOperator aggop, byte in1, byte in2, byte out, String istr)
	{
		super(null, in1, in2, out, istr);
		
		_uaggOp = uaggop;
		_aggOp = aggop;
		_bOp = bop;
			
		_tmpVal1 = new MatrixBlock();
		_tmpVal2 = new MatrixBlock();
		
		mrtype = MRINSTRUCTION_TYPE.UaggOuterChain;
		instString = istr;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Instruction parseInstruction( String str ) 
		throws DMLRuntimeException 
	{		
		//check number of fields (2/3 inputs, output, type)
		InstructionUtils.checkNumFields ( str, 5 );
		
		//parse instruction parts (without exec type)
		String[] parts = InstructionUtils.getInstructionParts( str );		
		
		AggregateUnaryOperator uaggop = InstructionUtils.parseBasicAggregateUnaryOperator(parts[1]);
		BinaryOperator bop = BinaryInstruction.parseBinaryOperator(parts[2]);
		byte in1 = Byte.parseByte(parts[3]);
		byte in2 = Byte.parseByte(parts[4]);
		byte out = Byte.parseByte(parts[5]);
		
		//derive aggregation operator from unary operator
		String aopcode = InstructionUtils.deriveAggregateOperatorOpcode(parts[1]);
		CorrectionLocationType corrLoc = InstructionUtils.deriveAggregateOperatorCorrectionLocation(parts[1]);
		String corrExists = (corrLoc != CorrectionLocationType.NONE) ? "true" : "false";
		AggregateOperator aop = InstructionUtils.parseAggregateOperator(aopcode, corrExists, corrLoc.toString());
	
		return new UaggOuterChainInstruction(bop, uaggop, aop, in1, in2, out, str);
	}
	
	/**
	 * 
	 * @param mcIn
	 * @param mcOut
	 */
	public void computeOutputCharacteristics(MatrixCharacteristics mcIn, MatrixCharacteristics mcOut)
	{
		if( _uaggOp.indexFn instanceof ReduceAll )
			mcOut.set(1, 1, mcIn.getRowsPerBlock(), mcOut.getColsPerBlock());
		else if( _uaggOp.indexFn instanceof ReduceCol ) //e.g., rowSums
			mcOut.set(mcIn.getRows(), 1, mcIn.getRowsPerBlock(), mcOut.getColsPerBlock());
		else if( _uaggOp.indexFn instanceof ReduceRow ) //e.g., colSums
			mcOut.set(1, mcIn.getCols(), mcIn.getRowsPerBlock(), mcOut.getColsPerBlock());
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass, CachedValueMap cachedValues, 
			           IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get( input1 );
		if( blkList == null )
			return;

		for(IndexedMatrixValue imv : blkList)
		{
			if(imv == null)
				continue;
			
			MatrixIndexes in1Ix = imv.getIndexes();
			MatrixValue in1Val = imv.getValue();
			
			//allocate space for the intermediate and output value
			IndexedMatrixValue iout = cachedValues.holdPlace(output, valueClass);
			MatrixIndexes outIx = iout.getIndexes();
			MatrixValue outVal = iout.getValue();
			MatrixBlock corr = null;
			
			//process instruction
			DistributedCacheInput dcInput = MRBaseForCommonInstructions.dcValues.get(input2);
			
			if(    _bOp.fn instanceof LessThan && _uaggOp.aggOp.increOp.fn instanceof KahanPlus
				&& _uaggOp.indexFn instanceof ReduceCol ) //special case: rowsSums(outer(A,B,<))
			{
				//approach: for each ai, do binary search in B, position gives counts
				//step 1: prepare sorted rhs input (once per task)
				if( _bv == null ) {
					_bv = dcInput.getRowVectorArray();
					Arrays.sort(_bv);
				}
				
				//step2: prepare output (last col corr)
				outIx.setIndexes(in1Ix.getRowIndex(), 1); 
				outVal.reset(in1Val.getNumRows(), 2, false);
				
				//step3: compute unary aggregate outer chain
				MatrixBlock a = (MatrixBlock)in1Val;
				MatrixBlock c = (MatrixBlock)outVal;
				for( int i=0; i<a.getNumRows(); i++ ) {
					double ai = a.quickGetValue(i, 0);
					int ix = Arrays.binarySearch(_bv, ai);
					if( ix >= 0 ){ //match, scan to next val
						while( ai==_bv[ix++] && ix<_bv.length );
						ix += (ai==_bv[_bv.length-1])?1:0;
					}
					int cnt = _bv.length-Math.abs(ix)+1;
					c.quickSetValue(i, 0, cnt);
				}
			}
			else //default case 
			{
				long in2_cols = dcInput.getNumCols();
				long  in2_colBlocks = (long)Math.ceil(((double)in2_cols)/dcInput.getNumColsPerBlock());
				
				for(int bidx=1; bidx <= in2_colBlocks; bidx++) 
				{
					IndexedMatrixValue imv2 = dcInput.getDataBlock(1, bidx);
					MatrixValue in2Val = imv2.getValue(); 
					
					//outer block operation
					OperationsOnMatrixValues.performBinaryIgnoreIndexes(in1Val, in2Val, _tmpVal1, _bOp);
						
					//unary aggregate operation
					OperationsOnMatrixValues.performAggregateUnary( in1Ix, _tmpVal1, outIx, _tmpVal2, _uaggOp, blockRowFactor, blockColFactor);
					
					//aggregate over all rhs blocks
					if( corr == null ) {
						outVal.reset(_tmpVal2.getNumRows(), _tmpVal2.getNumColumns(), false);
						corr = new MatrixBlock(_tmpVal2.getNumRows(), _tmpVal2.getNumColumns(), false);
					}
					
					if(_aggOp.correctionExists)
						OperationsOnMatrixValues.incrementalAggregation(outVal, corr, _tmpVal2, _aggOp, true);
					else 
						OperationsOnMatrixValues.incrementalAggregation(outVal, null, _tmpVal2, _aggOp, true);
				}
			}
		}
	}
	
	public static boolean isDistCacheOnlyIndex( String inst, byte index )
	{
		boolean ret = false;
		
		//parse instruction parts (with exec type)
		String[] parts = inst.split(Instruction.OPERAND_DELIM);
		byte in1 = Byte.parseByte(parts[4].split(Instruction.DATATYPE_PREFIX)[0]);
		byte in2 = Byte.parseByte(parts[5].split(Instruction.DATATYPE_PREFIX)[0]);
		ret = (index==in2 && index!=in1);
		
		return ret;
	}
	
	public static void addDistCacheIndex( String inst, ArrayList<Byte> indexes )
	{
		//parse instruction parts (with exec type)
		String[] parts = inst.split(Instruction.OPERAND_DELIM);
		byte in2 = Byte.parseByte(parts[5].split(Instruction.DATATYPE_PREFIX)[0]);
		indexes.add(in2);
	}
}
