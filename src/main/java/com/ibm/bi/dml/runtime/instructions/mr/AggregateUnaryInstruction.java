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
import com.ibm.bi.dml.runtime.functionobjects.ReduceDiag;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class AggregateUnaryInstruction extends UnaryMRInstructionBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private boolean _dropCorr = false;
	
	public AggregateUnaryInstruction(Operator op, byte in, byte out, boolean dropCorr, String istr)
	{
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.AggregateUnary;
		instString = istr;
		
		_dropCorr = dropCorr;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 3 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		String opcode = parts[0];
		byte in = Byte.parseByte(parts[1]);
		byte out = Byte.parseByte(parts[2]);
		boolean drop = Boolean.parseBoolean(parts[3]);
		
		AggregateUnaryOperator aggun = InstructionUtils.parseBasicAggregateUnaryOperator(opcode);
		return new AggregateUnaryInstruction(aggun, in, out, drop, str);
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, 
			int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(input);
		if( blkList != null )
			for(IndexedMatrixValue in: blkList)
			{
				if(in==null)
					continue;
			
				//allocate space for the output value
				IndexedMatrixValue out;
				if(input==output)
					out=tempValue;
				else
					out=cachedValues.holdPlace(output, valueClass);
				
				MatrixIndexes inix = in.getIndexes();
				
				//prune unnecessary blocks for trace
				if( (((AggregateUnaryOperator)optr).indexFn instanceof ReduceDiag && inix.getColumnIndex()!=inix.getRowIndex()) )
				{
					//do nothing (block not on diagonal); but reset
					out.getValue().reset();
				}
				else //general case
				{
					//process instruction
					AggregateUnaryOperator auop = (AggregateUnaryOperator)optr;
					OperationsOnMatrixValues.performAggregateUnary( inix, in.getValue(), out.getIndexes(), out.getValue(), 
							                            auop, blockRowFactor, blockColFactor);
					if( _dropCorr )
						((MatrixBlock)out.getValue()).dropLastRowsOrColums(auop.aggOp.correctionLocation);
				}
				
				//put the output value in the cache
				if(out==tempValue)
					cachedValues.add(output, out);
			}
	}

}
