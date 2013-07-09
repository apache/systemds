package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
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
	public void processInstruction(ExecutionContext ec)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		MatrixBlock matBlock = (MatrixBlock) ec.getMatrixInput(input1.get_name());
		MatrixBlock wtBlock = null, resultBlock = null;
 		
		String output_name = output.get_name();
		
		if (input2 != null) {
			wtBlock = (MatrixBlock) ec.getMatrixInput(input2.get_name());
		}
		
		resultBlock = (MatrixBlock) matBlock.sortOperations(wtBlock, new MatrixBlock());
		
		matBlock = wtBlock = null;
		ec.releaseMatrixInput(input1.get_name());
		if (input2 != null)
			ec.releaseMatrixInput(input2.get_name());
		
		ec.setMatrixOutput(output_name, resultBlock);
	}
}
