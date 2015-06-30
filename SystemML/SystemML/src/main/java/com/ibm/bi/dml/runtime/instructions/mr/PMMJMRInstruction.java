/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.mr;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.PMMJ.CacheType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.DistributedCacheInput;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.MRBaseForCommonInstructions;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.util.UtilFunctions;


/**
 * 
 * 
 */
public class PMMJMRInstruction extends BinaryMRInstructionBase
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private long _rlen = -1;
	private boolean _outputEmptyBlocks = true;
	
	
	public PMMJMRInstruction(Operator op, byte in1, byte in2, byte out, long nrow, CacheType ctype, boolean outputEmpty, String istr)
	{
		super(op, in1, in2, out);
		instString = istr;
		
		_rlen = nrow;
		_outputEmptyBlocks = outputEmpty;
		
		//NOTE: cache type only used by distributed cache input
	}
	
	public long getNumRows() {
		return _rlen;
	}
	
	public boolean getOutputEmptyBlocks() {
		return _outputEmptyBlocks;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		InstructionUtils.checkNumFields ( str, 6 );
		
		String[] parts = InstructionUtils.getInstructionParts(str);
		String opcode = parts[0];
		byte in1 = Byte.parseByte(parts[1]);
		byte in2 = Byte.parseByte(parts[2]);
		byte out = Byte.parseByte(parts[4]);
		long nrow = Long.parseLong(parts[3]);
		CacheType ctype = CacheType.valueOf(parts[5]);
		boolean outputEmpty = Boolean.parseBoolean(parts[6]);
		
		if(!opcode.equalsIgnoreCase("pmm"))
			throw new DMLRuntimeException("Unknown opcode while parsing an PmmMRInstruction: " + str);
		
		return new PMMJMRInstruction(new Operator(true), in1, in2, out, nrow, ctype, outputEmpty, str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		//get both matrix inputs (left side always permutation)
		DistributedCacheInput dcInput = MRBaseForCommonInstructions.dcValues.get(input1);
		IndexedMatrixValue in2 = cachedValues.getFirst(input2);
		IndexedMatrixValue in1 = dcInput.getDataBlock((int)in2.getIndexes().getRowIndex(), 1);
		MatrixBlock mb1 = (MatrixBlock)in1.getValue();
		MatrixBlock mb2 = (MatrixBlock)in2.getValue();
		
		//compute target block indexes
		long minPos = UtilFunctions.toLong( mb1.minNonZero() );
		long maxPos = UtilFunctions.toLong( mb1.max() );
		long rowIX1 = (minPos-1)/blockRowFactor+1;
		long rowIX2 = (maxPos-1)/blockRowFactor+1;
		boolean multipleOuts = (rowIX1 != rowIX2);
		
		if( minPos >= 1 ) //at least one row selected
		{
			//output sparsity estimate
			double spmb1 = OptimizerUtils.getSparsity(mb1.getNumRows(), 1, mb1.getNonZeros());
			long estnnz = (long) (spmb1 * mb2.getNonZeros());
			boolean sparse = MatrixBlock.evalSparseFormatInMemory(blockRowFactor, mb2.getNumColumns(), estnnz);
			
			//compute and allocate output blocks
			IndexedMatrixValue out1 = cachedValues.holdPlace(output, valueClass);
			IndexedMatrixValue out2 = multipleOuts ? cachedValues.holdPlace(output, valueClass) : null;
			out1.getValue().reset(blockRowFactor, mb2.getNumColumns(), sparse);
			if( out2 != null )
				out2.getValue().reset(UtilFunctions.computeBlockSize(_rlen, rowIX2, blockRowFactor), mb2.getNumColumns(), sparse);
			
			//compute core matrix permutation (assumes that out1 has default blocksize, 
			//hence we do a meta data correction afterwards)
			mb1.permutationMatrixMultOperations(mb2, out1.getValue(), (out2!=null)?out2.getValue():null);
			((MatrixBlock)out1.getValue()).setNumRows(UtilFunctions.computeBlockSize(_rlen, rowIX1, blockRowFactor));
			out1.getIndexes().setIndexes(rowIX1, in2.getIndexes().getColumnIndex());
			if( out2 != null )
				out2.getIndexes().setIndexes(rowIX2, in2.getIndexes().getColumnIndex());
				
			//empty block output filter (enabled by compiler consumer operation is in CP)
			if( !_outputEmptyBlocks && out1.getValue().isEmpty() 
				&& (out2==null || out2.getValue().isEmpty() )  )
			{
				cachedValues.remove(output);
			}
		}
	}
	
	/**
	 * Determines if the given index is only used via distributed cache in
	 * the given instruction string (used during setup of distributed cache
	 * to detect redundant job inputs).
	 * 
	 * @param inst
	 * @param index
	 * @return
	 */
	public static boolean isDistCacheOnlyIndex( String inst, byte index )
	{
		//parse instruction parts (with exec type)
		String[] parts = inst.split(Instruction.OPERAND_DELIM);
		byte in1 = Byte.parseByte(parts[2].split(Instruction.DATATYPE_PREFIX)[0]);
		byte in2 = Byte.parseByte(parts[3].split(Instruction.DATATYPE_PREFIX)[0]);
		return (index==in1 && index!=in2);
	}
	
	/**
	 * 
	 * @param inst
	 * @param indexes
	 */
	public static void addDistCacheIndex( String inst, ArrayList<Byte> indexes )
	{
		//parse instruction parts (with exec type)
		String[] parts = inst.split(Instruction.OPERAND_DELIM);
		byte in1 = Byte.parseByte(parts[2].split(Instruction.DATATYPE_PREFIX)[0]);
		indexes.add(in1);
	}
}
