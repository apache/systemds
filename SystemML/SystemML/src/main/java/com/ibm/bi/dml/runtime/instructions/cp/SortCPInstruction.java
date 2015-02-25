/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;

public class SortCPInstruction extends UnaryCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	/*
	 * This class supports two variants of sort operation on a 1-dimensional input matrix. 
	 * The two variants are <code> weighted </code> and <code> unweighted </code>.
	 * Example instructions: 
	 *     sort:mVar1:mVar2 (input=mVar1, output=mVar2)
	 *     sort:mVar1:mVar2:mVar3 (input=mVar1, weights=mVar2, output=mVar3)
	 *  
	 */
	
	public SortCPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String istr){
		this(op, in, null, out, opcode, istr);
	}
	
	public SortCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr){
		super(op, in1, in2, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.Sort;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = null;
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase("sort") ) {
			if ( parts.length == 3 ) {
				// Example: sort:mVar1:mVar2 (input=mVar1, output=mVar2)
				parseUnaryInstruction(str, in1, out);
				return new SortCPInstruction(new SimpleOperator(null), in1, out, opcode, str);
			}
			else if ( parts.length == 4 ) {
				// Example: sort:mVar1:mVar2:mVar3 (input=mVar1, weights=mVar2, output=mVar3)
				in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
				parseUnaryInstruction(str, in1, in2, out);
				return new SortCPInstruction(new SimpleOperator(null), in1, in2, out, opcode, str);
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
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		//acquire inputs matrices
		MatrixBlock matBlock = ec.getMatrixInput(input1.getName());
		MatrixBlock wtBlock = null;
 		if (input2 != null) {
			wtBlock = ec.getMatrixInput(input2.getName());
		}
		
 		//process core instruction
		MatrixBlock resultBlock = (MatrixBlock) matBlock.sortOperations(wtBlock, new MatrixBlock());
		
		//release inputs
		ec.releaseMatrixInput(input1.getName());
		if (input2 != null)
			ec.releaseMatrixInput(input2.getName());
		
		//set and release output
		ec.setMatrixOutput(output.getName(), resultBlock);
	}
}
