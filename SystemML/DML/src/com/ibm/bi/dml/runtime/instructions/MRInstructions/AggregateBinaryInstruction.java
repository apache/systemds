package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.MRBaseForCommonInstructions;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class AggregateBinaryInstruction extends BinaryMRInstructionBase {

	public AggregateBinaryInstruction(Operator op, byte in1, byte in2, byte out, String istr)
	{
		super(op, in1, in2, out);
		mrtype = MRINSTRUCTION_TYPE.AggregateBinary;
		instString = istr;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 3 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in1, in2, out;
		String opcode = parts[0];
		in1 = Byte.parseByte(parts[1]);
		in2 = Byte.parseByte(parts[2]);
		out = Byte.parseByte(parts[3]);
		
		if ( opcode.equalsIgnoreCase("cpmm") || opcode.equalsIgnoreCase("rmm") || opcode.equalsIgnoreCase("mvmult") ) {
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateBinaryOperator aggbin = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);
			return new AggregateBinaryInstruction(aggbin, in1, in2, out, str);
		} 
		else {
			throw new DMLRuntimeException("AggregateBinaryInstruction.parseInstruction():: Unknown opcode " + opcode);
		}
		
		// return null;
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		IndexedMatrixValue in1=cachedValues.getFirst(input1);
		IndexedMatrixValue in2=cachedValues.getFirst(input2);

		MatrixValue vector=null;
		if ( in2 == null ) {
			//vector = MRBaseForCommonInstructions.loadDataFromDistributedCache(input2, in1.getIndexes().getColumnIndex(), 1); // MRBaseForCommonInstructions.distCacheValues.get(input2);
			vector = MRBaseForCommonInstructions.readBlockFromDistributedCache(input2, in1.getIndexes().getColumnIndex(), 1, blockRowFactor, blockColFactor); // MRBaseForCommonInstructions.distCacheValues.get(input2);
			if ( vector == null )
				throw new DMLRuntimeException("Unexpected: vector read from distcache is null!");
			in2 = new IndexedMatrixValue(new MatrixIndexes(in1.getIndexes().getColumnIndex(),1), vector);
		}
		
		if(in1==null || (in2==null && vector==null))
			return;

		//allocate space for the output value
		IndexedMatrixValue out;
		if(output==input1 || output==input2)
			out=tempValue;
		else
			out=cachedValues.holdPlace(output, valueClass);
		
		//process instruction
		//if ( instString.contains("mvmult") ) {
		//	OperationsOnMatrixValues.performAggregateBinary(in1.getIndexes(), in1.getValue(), 
		//			new MatrixIndexes(1,1), vector, out.getIndexes(), out.getValue(), 
		//			((AggregateBinaryOperator)optr), true);
		//}
		//else {
			//System.out.println("matmult: [" + in1.getIndexes() + "] x [" + in2.getIndexes() +"]");
			OperationsOnMatrixValues.performAggregateBinary(in1.getIndexes(), in1.getValue(), 
					in2.getIndexes(), in2.getValue(), out.getIndexes(), out.getValue(), 
					((AggregateBinaryOperator)optr), false);
		//}
		
		//put the output value in the cache
		if(out==tempValue)
			cachedValues.add(output, out);
		//System.out.println("--> " + in1.getIndexes() + " x " + in2.getIndexes() + " = " + out.getIndexes());
	}

}
