package dml.runtime.instructions.CPInstructions;

import dml.parser.Expression.DataType;
import dml.parser.Expression.ValueType;
import dml.runtime.functionobjects.MaxIndex;
import dml.runtime.functionobjects.SwapIndex;
import dml.runtime.instructions.Instruction;
import dml.runtime.controlprogram.ProgramBlock;
import dml.runtime.matrix.operators.Operator;
import dml.runtime.matrix.operators.ReorgOperator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class ReorgCPInstruction extends UnaryCPInstruction{
	public ReorgCPInstruction(Operator op, CPOperand in, CPOperand out, String istr){
		super(op, in, out, istr);
		cptype = CPINSTRUCTION_TYPE.Reorg;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = parseUnaryInstruction(str, in, out);
		
		if ( opcode.equalsIgnoreCase("r'") ) {
			return new ReorgCPInstruction(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), in, out, str);
		} 
		
		else if ( opcode.equalsIgnoreCase("rdiagV2M") ) {
			return new ReorgCPInstruction(new ReorgOperator(MaxIndex.getMaxIndexFnObject()), in, out, str);
		} 
		
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a ReorgInstruction: " + str);
		}
	}
	
	@Override
	public MatrixObject processInstruction(ProgramBlock pb)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		MatrixObject mat = pb.getMatrixVariable(input1.get_name());
		
		ReorgOperator r_op = (ReorgOperator) optr;
		
		String output_name = output.get_name();
		MatrixObject sores = mat.reorgOperations(r_op, (MatrixObject)pb.getVariable(output_name));
		
		pb.setVariableAndWriteToHDFS(output_name, sores);
		return sores; 
	}
}
