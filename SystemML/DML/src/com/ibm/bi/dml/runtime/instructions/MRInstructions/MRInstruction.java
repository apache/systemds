package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public abstract class MRInstruction extends Instruction {

	public enum MRINSTRUCTION_TYPE { INVALID, Append, Aggregate, ArithmeticBinary, AggregateBinary, AggregateUnary, Rand, 
		Reblock, Reorg, Replicate, Unary, CombineBinary, CombineUnary, CombineTertiary, PickByCount, 
		Tertiary, CM_N_COV, Combine, GroupedAggregate, RangeReIndex, ZeroOut, MMTSJ }; 
	
	
	MRINSTRUCTION_TYPE mrtype;
	Operator optr;
	public byte output;
	
	public MRInstruction (Operator op, byte out) {
		type = INSTRUCTION_TYPE.MAPREDUCE;
		optr = op;
		output = out;
		mrtype = MRINSTRUCTION_TYPE.INVALID;
	}
	
	public Operator getOperator() {
		return optr;
	}
	
	public abstract void processInstruction(Class<? extends MatrixValue> valueClass, 
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput,
			int blockRowFactor, int blockColFactor) throws DMLUnsupportedOperationException, DMLRuntimeException;

	public MRINSTRUCTION_TYPE getMRInstructionType() 
	{
		return mrtype;
	}

	//public static MRInstruction parseMRInstruction ( String str ) throws DMLRuntimeException, DMLUnsupportedOperationException {
	//	return MRInstructionParser.parseSingleInstruction(str);
	//}
}
