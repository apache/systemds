/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import java.util.ArrayList;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.MaxIndex;
import com.ibm.bi.dml.runtime.functionobjects.SwapIndex;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;


public class ReorgInstruction extends UnaryMRInstructionBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//required for diag (load-balance-aware output of empty blocks)
	private MatrixCharacteristics _mcIn = null;
	private boolean _outputEmptyBlocks = true;
	
	public ReorgInstruction(Operator op, byte in, byte out, String istr)
	{
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.Reorg;
		instString = istr;
	}
	
	public void setInputMatrixCharacteristics( MatrixCharacteristics in )
	{
		_mcIn = in; 
	}
	
	public void setOutputEmptyBlocks( boolean flag )
	{
		_outputEmptyBlocks = flag; 
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 2 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in, out;
		String opcode = parts[0];
		in = Byte.parseByte(parts[1]);
		out = Byte.parseByte(parts[2]);
		
		if ( opcode.equalsIgnoreCase("r'") ) {
			return new ReorgInstruction(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("rdiagV2M") ) {
			return new ReorgInstruction(new ReorgOperator(MaxIndex.getMaxIndexFnObject()), in, out, str);
		} 
		
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a ReorgInstruction: " + str);
		}
		
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, 
			int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(input);
		
		if( blkList != null )
			for(IndexedMatrixValue in : blkList)
			{
				if(in==null)
					continue;
				int startRow=0, startColumn=0, length=0;
				
				//allocate space for the output value
				IndexedMatrixValue out;
				if(input==output)
					out=tempValue;
				else
					out=cachedValues.holdPlace(output, valueClass);
				
				//process instruction
				OperationsOnMatrixValues.performReorg(in.getIndexes(), in.getValue(), 
						out.getIndexes(), out.getValue(), ((ReorgOperator)optr),
						startRow, startColumn, length);
				
				//put the output value in the cache
				if(out==tempValue)
					cachedValues.add(output, out);
				
				//special handling for vector to matrix diag to make sure the missing 0' are accounted for 
				//(only for block representation)
				if(_outputEmptyBlocks && valueClass.equals(MatrixBlock.class) && ((ReorgOperator)optr).fn==MaxIndex.getMaxIndexFnObject())
				{
					long diagIndex=out.getIndexes().getRowIndex();//row index is equal to the col index
					long rlen = Math.max(_mcIn.get_rows(), _mcIn.get_cols()); //input can be row/column vector
					long brlen = Math.max(_mcIn.get_rows_per_block(),_mcIn.get_cols_per_block());
					long numRowBlocks = (rlen/brlen)+((rlen%brlen!=0)? 1 : 0);
					for(long rc=1; rc<=numRowBlocks; rc++)
					{
						if( rc==diagIndex ) continue; //prevent duplicate output
						IndexedMatrixValue emptyIndexValue=cachedValues.holdPlace(output, valueClass);
						int lbrlen = (int) ((rc*brlen<=rlen) ? brlen : rlen%brlen);
						emptyIndexValue.getIndexes().setIndexes(rc, diagIndex);
						emptyIndexValue.getValue().reset(lbrlen, out.getValue().getNumColumns(), true);
					}
					
					// MB> this code led to huge imbalance because mappers with high row index created
					//     many blocks (e.g., ix=3->4 vs ix=100,000->200,000 blocks ). 					
					/*
					for(long rc=1; rc<diagIndex; rc++)
					{
						IndexedMatrixValue emptyIndexValue=cachedValues.holdPlace(output, valueClass);
						emptyIndexValue.getIndexes().setIndexes(rc, diagIndex);
						emptyIndexValue.getValue().reset(blockRowFactor, out.getValue().getNumColumns(), true);
						emptyIndexValue=cachedValues.holdPlace(output, valueClass);
						emptyIndexValue.getIndexes().setIndexes(diagIndex, rc);
						emptyIndexValue.getValue().reset(out.getValue().getNumRows(), blockColFactor, true);
					}
					*/
				}
			}
	}
}
