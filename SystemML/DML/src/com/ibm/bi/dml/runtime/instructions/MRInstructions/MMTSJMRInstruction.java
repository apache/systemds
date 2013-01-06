package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import com.ibm.bi.dml.lops.MMTSJ.MMTSJType;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


/**
 * 
 * 
 */
public class MMTSJMRInstruction extends UnaryInstruction
{	
	private MMTSJType _type = null;

	public MMTSJMRInstruction(Operator op, byte in, MMTSJType type, byte out, String istr)
	{
		super(op, in, out, istr);
		mrtype = MRINSTRUCTION_TYPE.MMTSJ;
		instString = istr;
		
		_type = type;
	}
	
	/**
	 * 
	 * @return
	 */
	public MMTSJType getMMTSJType()
	{
		return _type;
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
		InstructionUtils.checkNumFields ( str, 3 );
		
		String[] parts = InstructionUtils.getInstructionParts(str);
		String opcode = parts[0];
		byte in = Byte.parseByte(parts[1]);
		byte out = Byte.parseByte(parts[2]);
		MMTSJType titype = MMTSJType.valueOf(parts[3]);
		 
		if(!opcode.equalsIgnoreCase("tsmm"))
			throw new DMLRuntimeException("Unknown opcode while parsing an MMTIJMRInstruction: " + str);
		else
			return new MMTSJMRInstruction(new Operator(true), in, titype, out, str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{		
		MatrixValue in = cachedValues.getFirst(input).getValue();
		
		//allocate space for the output value
		IndexedMatrixValue iout = null;
		if(output==input)
			iout=tempValue;
		else
			iout=cachedValues.holdPlace(output, valueClass);
		iout.getIndexes().setIndexes(1, 1);
		MatrixValue out = iout.getValue();
		
		//process instruction
		if( in instanceof MatrixBlock && out instanceof MatrixBlock )
			((MatrixBlock) in).transposeSelfMatrixMult((MatrixBlock)out, _type );
		else
			throw new DMLUnsupportedOperationException("Types "+in.getClass()+" and "+out.getClass()+" incompatible with "+MatrixBlock.class);
		
		//put the output value in the cache
		if(iout==tempValue)
			cachedValues.add(output, iout);
	}
}
