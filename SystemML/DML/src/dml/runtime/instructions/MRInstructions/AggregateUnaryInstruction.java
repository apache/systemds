package dml.runtime.instructions.MRInstructions;

import dml.lops.PartialAggregate.CorrectionLocationType;
import dml.runtime.functionobjects.Builtin;
import dml.runtime.functionobjects.KahanPlus;
import dml.runtime.functionobjects.Multiply;
import dml.runtime.functionobjects.Plus;
import dml.runtime.functionobjects.ReduceAll;
import dml.runtime.functionobjects.ReduceCol;
import dml.runtime.functionobjects.ReduceRow;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.OperationsOnMatrixValues;
import dml.runtime.matrix.mapred.CachedValueMap;
import dml.runtime.matrix.mapred.IndexedMatrixValue;
import dml.runtime.matrix.operators.AggregateOperator;
import dml.runtime.matrix.operators.AggregateUnaryOperator;
import dml.runtime.matrix.operators.Operator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class AggregateUnaryInstruction extends UnaryMRInstructionBase {

	public AggregateUnaryInstruction(Operator op, byte in, byte out, String istr)
	{
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.AggregateUnary;
		instString = istr;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 2 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in, out;
		String opcode = parts[0];
		in = Byte.parseByte(parts[1]);
		out = Byte.parseByte(parts[2]);
		
		if ( opcode.equalsIgnoreCase("uak+") ) {
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTCOLUMN);
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
			return new AggregateUnaryInstruction(aggun, in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uark+") ) {
			// RowSums
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTCOLUMN);
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
			return new AggregateUnaryInstruction(aggun, in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uack+") ) {
			// ColSums
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTROW);
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
			return new AggregateUnaryInstruction(aggun, in, out, str);
		}
		else if ( opcode.equalsIgnoreCase("uamean") ) {
			// Mean
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTTWOCOLUMNS);
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
			return new AggregateUnaryInstruction(aggun, in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uarmean") ) {
			// RowMeans
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTTWOCOLUMNS);
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
			return new AggregateUnaryInstruction(aggun, in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uarimax") ) {
			// returns col index of max in row
			AggregateOperator agg = new AggregateOperator(0, Builtin.getBuiltinFnObject("maxindex"), true, CorrectionLocationType.LASTCOLUMN);
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
			return new AggregateUnaryInstruction(aggun, in, out, str);
		}
		
		else if ( opcode.equalsIgnoreCase("uacmean") ) {
			// ColMeans
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTTWOROWS);
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
			return new AggregateUnaryInstruction(aggun, in, out, str);
		}
		else if ( opcode.equalsIgnoreCase("ua+") ) {
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
			return new AggregateUnaryInstruction(aggun, in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uar+") ) {
			// RowSums
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
			return new AggregateUnaryInstruction(aggun, in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uac+") ) {
			// ColSums
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
			return new AggregateUnaryInstruction(aggun, in, out, str);
		}
		
		else if ( opcode.equalsIgnoreCase("ua*") ) {
			AggregateOperator agg = new AggregateOperator(1, Multiply.getMultiplyFnObject());
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
			return new AggregateUnaryInstruction(aggun, in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uamax") ) {
			AggregateOperator agg = new AggregateOperator(Double.MIN_VALUE, Builtin.getBuiltinFnObject("max"));
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
			return new AggregateUnaryInstruction(aggun, in, out, str);
			// return new AggregateUnaryInstruction(new BinaryOperator(Builtin.getBuiltinFnObject("max")), in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uamin") ) {
			AggregateOperator agg = new AggregateOperator(Double.MAX_VALUE, Builtin.getBuiltinFnObject("min"));
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
			return new AggregateUnaryInstruction(aggun, in, out, str);
			// return new AggregateUnaryInstruction(new BinaryOperator(Builtin.getBuiltinFnObject("min")), in, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("uatrace") ) {
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
			aggun.isTrace=true;
			return new AggregateUnaryInstruction(aggun, in, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("uaktrace") ) {
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTCOLUMN);
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject());
			aggun.isTrace=true;
			return new AggregateUnaryInstruction(aggun, in, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("rdiagM2V") ) {
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
			aggun.isDiagM2V=true;
			return new AggregateUnaryInstruction(aggun, in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uarmax") ) {
			AggregateOperator agg = new AggregateOperator(0, Builtin.getBuiltinFnObject("max"));
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
			return new AggregateUnaryInstruction(aggun, in, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("uarmin") ) {
			AggregateOperator agg = new AggregateOperator(0, Builtin.getBuiltinFnObject("min"));
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject());
			return new AggregateUnaryInstruction(aggun, in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("uacmax") ) {
			AggregateOperator agg = new AggregateOperator(0, Builtin.getBuiltinFnObject("max"));
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
			return new AggregateUnaryInstruction(aggun, in, out, str);
		} 
		else if ( opcode.equalsIgnoreCase("uacmin") ) {
			AggregateOperator agg = new AggregateOperator(0, Builtin.getBuiltinFnObject("min"));
			AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject());
			return new AggregateUnaryInstruction(aggun, in, out, str);
		} 
		
		return null;
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, 
			int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		IndexedMatrixValue in=cachedValues.getFirst(input);
		if(in==null)
			return;
		
		//allocate space for the output value
		IndexedMatrixValue out;
		if(input==output)
			out=tempValue;
		else
			out=cachedValues.holdPlace(output, valueClass);
		
	/*	String opcode = InstructionUtils.getOpCode(instString);
		// TODO: hack to support trace. Need to remove it in future.
		if ( opcode.equals("uatrace") || opcode.equals("uaktrace") || opcode.equals("rdiagM2V") )
			brlen = bclen = DMLTranslator.DMLBlockSize;
	*/	
		//process instruction
		OperationsOnMatrixValues.performAggregateUnary(in.getIndexes(), in.getValue(), 
				out.getIndexes(), out.getValue(), ((AggregateUnaryOperator)optr), blockRowFactor, blockColFactor);
		
		//put the output value in the cache
		if(out==tempValue)
			cachedValues.add(output, out);
	}

}
