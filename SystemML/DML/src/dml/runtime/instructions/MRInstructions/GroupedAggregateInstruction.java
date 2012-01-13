package dml.runtime.instructions.MRInstructions;

import dml.runtime.functionobjects.CM;
import dml.runtime.functionobjects.KahanPlus;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.mapred.CachedValueMap;
import dml.runtime.matrix.mapred.IndexedMatrixValue;
import dml.runtime.matrix.operators.AggregateOperator;
import dml.runtime.matrix.operators.CMOperator;
import dml.runtime.matrix.operators.Operator;
import dml.runtime.matrix.operators.CMOperator.AggregateOperationTypes;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class GroupedAggregateInstruction extends UnaryMRInstructionBase{

	
	public GroupedAggregateInstruction(Operator op, byte in, byte out, String istr) {
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.GroupedAggregate;
		instString = istr;
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput,
			int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new DMLRuntimeException("GroupedAggregateInstruction.processInstruction() should not be called!");
		
	}

	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		if(parts.length<2)
			throw new DMLRuntimeException("the number of fields of instruction "+str+" is less than 2!");
		byte in, out;
		String opcode = parts[0];
		in = Byte.parseByte(parts[1]);
		out = Byte.parseByte(parts[parts.length - 1]);
		
		if ( !opcode.equalsIgnoreCase("groupedagg") ) {
			throw new DMLRuntimeException("Invalid opcode in GroupedAggregateInstruction: " + opcode);
		}
		
		// parts[2] should point to the function
		AggregateOperationTypes op = AggregateOperationTypes.INVALID;
		if ( parts[2].equalsIgnoreCase("centralmoment") )
			// in case of CM, we also need to pass "order"
			op = CMOperator.getAggOpType(parts[2], parts[3]);
		else 
			op = CMOperator.getAggOpType(parts[2], null);
		
		switch(op) {
		case SUM:
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, (byte)2);
			return new GroupedAggregateInstruction(agg, in, out, str);
			
		case COUNT:
		case MEAN:
		case VARIANCE:
		case CM2:
		case CM3:
		case CM4:
			CMOperator cm = new CMOperator(CM.getCMFnObject(), op);
			return new GroupedAggregateInstruction(cm, in, out, str);
		case INVALID:
		default:
			throw new DMLRuntimeException("Invalid Aggregate Operation in GroupedAggregateInstruction: " + op);
		}
		
		/*
		if ( opcode.equalsIgnoreCase("grpak+") ) {
		} 
		
		else if ( opcode.equalsIgnoreCase("grpcm") ) {
			// RowSums
			if(parts.length<3)
				throw new DMLRuntimeException("the number of fields of instruction "+str+" is less than 3!");
			int cst=Integer.parseInt(parts[3]);
			if(cst>4 || cst<0)
				throw new DMLRuntimeException("constant for central moment has to be 0<= <5");
			
			CMOperator cm = new CMOperator(CM.getCMFnObject(), cst);
			return new GroupedAggregateInstruction(cm, in, out, str);
		} 		
		return null;
		*/
	}

}
