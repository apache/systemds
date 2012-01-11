package dml.runtime.instructions.MRInstructions;

import dml.runtime.functionobjects.Builtin;
import dml.runtime.functionobjects.KahanPlus;
import dml.runtime.functionobjects.Multiply;
import dml.runtime.functionobjects.Plus;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.mapred.CachedValueMap;
import dml.runtime.matrix.mapred.IndexedMatrixValue;
import dml.runtime.matrix.operators.AggregateOperator;
import dml.runtime.matrix.operators.Operator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class AggregateInstruction extends UnaryMRInstructionBase {
	
	public AggregateInstruction(Operator op, byte in, byte out, String istr)
	{
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.Aggregate;
		instString = istr;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in, out;
		String opcode = parts[0];
		in = Byte.parseByte(parts[1]);
		out = Byte.parseByte(parts[2]);
		
		if(opcode.equalsIgnoreCase("ak+") || opcode.equalsIgnoreCase("amean"))
			InstructionUtils.checkNumFields ( str, 4 );
		else
			InstructionUtils.checkNumFields ( str, 2 );
		
		if ( opcode.equalsIgnoreCase("ak+") ) {
			boolean corExists=Boolean.parseBoolean(parts[3]);
			byte loc=Byte.parseByte(parts[4]);
			
			// if corrections are not available, then we must use simple sum
			AggregateOperator agg = null; 
			if ( corExists ) {
				agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), corExists, loc);
			}
			else {
				agg = new AggregateOperator(0, Plus.getPlusFnObject(), corExists, loc);
			}
			return new AggregateInstruction(agg, in, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("a+") ) {
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			return new AggregateInstruction(agg, in, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("a*") ) {
			AggregateOperator agg = new AggregateOperator(1, Multiply.getMultiplyFnObject());
			return new AggregateInstruction(agg, in, out, str);
		}
		else if ( opcode.equalsIgnoreCase("amax") ) {
			AggregateOperator agg = new AggregateOperator(Double.MIN_VALUE, Builtin.getBuiltinFnObject("max"));
			return new AggregateInstruction(agg, in, out, str);
		}
		else if ( opcode.equalsIgnoreCase("amin") ) {
			AggregateOperator agg = new AggregateOperator(Double.MAX_VALUE, Builtin.getBuiltinFnObject("min"));
			return new AggregateInstruction(agg, in, out, str);
		}
		else if ( opcode.equalsIgnoreCase("amean") ) {
			// example: amean°1·DOUBLE°4·DOUBLE°true°4
			boolean corExists=Boolean.parseBoolean(parts[3]);
			byte loc=Byte.parseByte(parts[4]);
			
			// if corrections are not available, then we must use simple sum
			AggregateOperator agg = null; 
			if ( corExists ) {
				// stable mean internally makes use of Kahan summation
				agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), corExists, loc);
			}
			else {
				agg = new AggregateOperator(0, Plus.getPlusFnObject(), corExists, loc);
			}
			return new AggregateInstruction(agg, in, out, str);
		}
		return null;
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		throw new DMLRuntimeException("no processInstruction for AggregateInstruction!");
		
	}

}
