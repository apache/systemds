package dml.runtime.instructions.MRInstructions;

import dml.runtime.instructions.Instruction;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.mapred.CachedValueMap;
import dml.runtime.matrix.mapred.IndexedMatrixValue;
import dml.runtime.matrix.operators.Operator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public abstract class MRInstruction extends Instruction {

	public enum MRINSTRUCTION_TYPE { INVALID, Append, Aggregate, AggregateBinary, AggregateUnary, Binary, Rand, 
		Reblock, Reorg, Replicate, Scalar, Unary, CombineBinary, CombineUnary, CombineTertiary, PickByCount, 
		Tertiary, CM_N_COV, Combine, GroupedAggregate, RangeReIndex, Select }; 
	
	
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
