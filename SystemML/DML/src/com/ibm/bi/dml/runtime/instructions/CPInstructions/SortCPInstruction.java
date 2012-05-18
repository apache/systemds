package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;

public class SortCPInstruction extends UnaryCPInstruction{

	/*
	 * This class supports two variants of sort operation on a 1-dimensional input matrix. 
	 * The two variants are <code> weighted </code> and <code> unweighted </code>.
	 * Example instructions: 
	 *     sort:mVar1:mVar2 (input=mVar1, output=mVar2)
	 *     sort:mVar1:mVar2:mVar3 (input=mVar1, weights=mVar2, output=mVar3)
	 *  
	 */
	
	public SortCPInstruction(Operator op, CPOperand in, CPOperand out, String istr){
		this(op, in, null, out, istr);
	}
	
	public SortCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String istr){
		super(op, in1, in2, out, istr);
		cptype = CPINSTRUCTION_TYPE.Sort;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = null;
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		
		if ( parts[0].equalsIgnoreCase("sort") ) {
			if ( parts.length == 3 ) {
				// Example: sort:mVar1:mVar2 (input=mVar1, output=mVar2)
				parseUnaryInstruction(str, in1, out);
				return new SortCPInstruction(new SimpleOperator(null), in1, out, str);
			}
			else if ( parts.length == 4 ) {
				// Example: sort:mVar1:mVar2:mVar3 (input=mVar1, weights=mVar2, output=mVar3)
				in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
				parseUnaryInstruction(str, in1, in2, out);
				return new SortCPInstruction(new SimpleOperator(null), in1, in2, out, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		} 
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a SortCPInstruction: " + str);
		}
	}
	
	@Override
	public MatrixObject processInstruction(ProgramBlock pb)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		MatrixObject mat = pb.getMatrixVariable(input1.get_name());
		MatrixObject weights = null;
		if ( input2 != null ) {
			weights = pb.getMatrixVariable(input2.get_name());
		}
 		
		String output_name = output.get_name();
		MatrixObject sores = mat.sortOperations(weights, (MatrixObject)pb.getVariable(output_name));
		
		pb.setVariableAndWriteToHDFS(output_name, sores);
		return sores; 
	}
}
