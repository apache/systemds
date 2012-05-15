package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.lops.Tertiary;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class TertiaryCPInstruction extends ComputationCPInstruction{
	public TertiaryCPInstruction(Operator op, 
							 CPOperand in1, 
							 CPOperand in2, 
							 CPOperand in3, 
							 CPOperand out, 
						     String istr ){
		super(op, in1, in2, in3, out);
		instString = istr;
	}

	public static TertiaryCPInstruction parseInstruction(String inst) throws DMLRuntimeException{
		
		InstructionUtils.checkNumFields ( inst, 4 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(inst);
		String opcode = parts[0];
		
		if ( !opcode.equalsIgnoreCase("ctable") ) {
			throw new DMLRuntimeException("Unexpected opcode in TertiaryCPInstruction: " + inst);
		}
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		CPOperand out = new CPOperand(parts[4]);
		
		// ctable does not require any operator, so we simply pass-in a dummy operator with null functionobject
		return new TertiaryCPInstruction(new SimpleOperator(null), in1, in2, in3, out, inst);
	}
	
/*	static TertiaryOperator getTertiaryOperator(String opcode) throws DMLRuntimeException{
		throw new DMLRuntimeException("Unknown tertiary opcode " + opcode);
	}	
*/	

	private Tertiary.OperationTypes findCtableOperation() {
		DataType dt1 = input1.get_dataType();
		DataType dt2 = input2.get_dataType();
		DataType dt3 = input3.get_dataType();
		return Tertiary.findCtableOperationByInputDataTypes(dt1, dt2, dt3);
	}
	
	@Override
	public MatrixObject processInstruction(ProgramBlock pb) 
		throws DMLRuntimeException, DMLUnsupportedOperationException {
		
		MatrixObject in1 = pb.getMatrixVariable(input1.get_name());
		Data out = pb.getVariable(output.get_name());
		MatrixObject result = null;
		
		Tertiary.OperationTypes ctableOp = findCtableOperation();
		switch(ctableOp) {
		case CTABLE_TRANSFORM:
			// F=ctable(A,B,W)
			result = in1.tertiaryOperations((SimpleOperator)optr, pb.getMatrixVariable(input2.get_name()), pb.getMatrixVariable(input3.get_name()), (MatrixObject)out);
			break;
		case CTABLE_TRANSFORM_SCALAR_WEIGHT:
			// F = ctable(A,B) or F = ctable(A,B,1)
			result = in1.tertiaryOperations((SimpleOperator)optr, pb.getMatrixVariable(input2.get_name()), pb.getScalarVariable(input3.get_name(), ValueType.DOUBLE), (MatrixObject)out);
			break;
		case CTABLE_TRANSFORM_HISTOGRAM:
			// F=ctable(A,1) or F = ctable(A,1,1)
			result = in1.tertiaryOperations((SimpleOperator)optr, pb.getScalarVariable(input2.get_name(), ValueType.DOUBLE), pb.getScalarVariable(input3.get_name(), ValueType.DOUBLE), (MatrixObject)out);
			break;
		case CTABLE_TRANSFORM_WEIGHTED_HISTOGRAM:
			// F=ctable(A,1,W)
			result = in1.tertiaryOperations((SimpleOperator)optr, pb.getScalarVariable(input2.get_name(), ValueType.DOUBLE), pb.getMatrixVariable(input3.get_name()), (MatrixObject)out);
			break;
		
		default:
			throw new DMLRuntimeException("Encountered an invalid ctable operation ("+ctableOp+") while executing instruction: " + this.toString());
		}
		
		pb.setVariableAndWriteToHDFS(output.get_name(), result);
		return result;
	}

	
}
