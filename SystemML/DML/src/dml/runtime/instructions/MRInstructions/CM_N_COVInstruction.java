package dml.runtime.instructions.MRInstructions;

import dml.runtime.functionobjects.CM;
import dml.runtime.functionobjects.COV;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.mapred.CachedValueMap;
import dml.runtime.matrix.mapred.IndexedMatrixValue;
import dml.runtime.matrix.operators.CMOperator;
import dml.runtime.matrix.operators.COVOperator;
import dml.runtime.matrix.operators.Operator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class CM_N_COVInstruction extends UnaryMRInstructionBase {

	public CM_N_COVInstruction(Operator op, byte in, byte out, String istr)
	{
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.CM_N_COV;
		instString = istr;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in, out;
		int cst;
		String opcode = parts[0];
		
		if (opcode.equalsIgnoreCase("cm") ) 
		{
			in = Byte.parseByte(parts[1]);
			cst = Integer.parseInt(parts[2]);
			out = Byte.parseByte(parts[3]);
			
			if(cst>4 || cst<0 || cst==1)
				throw new DMLRuntimeException("constant for central moment has to be 0, 2, 3, or 4");
			
			CMOperator cm = new CMOperator(CM.getCMFnObject(), CMOperator.getCMAggOpType(cst));
			return new CM_N_COVInstruction(cm, in, out, str);
		}else if(opcode.equalsIgnoreCase("cov"))
		{
			in = Byte.parseByte(parts[1]);
			out = Byte.parseByte(parts[2]);
			COVOperator cov = new COVOperator(COV.getCOMFnObject());
			return new CM_N_COVInstruction(cov, in, out, str);
		}/*else if(opcode.equalsIgnoreCase("mean"))
		{
			in = Byte.parseByte(parts[1]);
			out = Byte.parseByte(parts[2]);
			
			CMOperator mean = new CMOperator(CM.getCMFnObject(), CMOperator.AggregateOperationTypes.MEAN);
			return new CM_N_COVInstruction(mean, in, out, str);
		}*/
		else
			throw new DMLRuntimeException("unknown opcode "+opcode);
		
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		throw new DMLRuntimeException("no processInstruction for AggregateInstruction!");
		
	}
}
