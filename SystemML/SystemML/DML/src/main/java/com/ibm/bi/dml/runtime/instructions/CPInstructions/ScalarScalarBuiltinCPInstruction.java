/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class ScalarScalarBuiltinCPInstruction extends BuiltinBinaryCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public ScalarScalarBuiltinCPInstruction(Operator op,
			  								CPOperand in1,
			  								CPOperand in2,
			  								CPOperand out,
			  								String instr){
		super(op, in1, in2, out, 2, instr);
	}

	@Override 
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException {
		
		String opcode = InstructionUtils.getOpCode(instString);
		ScalarObject sores = null;
		
		ScalarObject so1 = ec.getScalarInput( input1.get_name(), input1.get_valueType(), input1.isLiteral() );
		ScalarObject so2 = ec.getScalarInput(input2.get_name(), input2.get_valueType(), input2.isLiteral() );
		
		if ( opcode.equalsIgnoreCase("print") ) {
			String buffer = "";
			if (input2.get_valueType() != ValueType.STRING)
				throw new DMLRuntimeException("wrong value type in print");
			buffer = so2.getStringValue() + " ";
			
			if ( !DMLScript.suppressPrint2Stdout()) {
				switch (input1.get_valueType()) {
				case INT:
					System.out.println(buffer + so1.getLongValue());
					break;
				case DOUBLE:
					System.out.println(buffer + so1.getDoubleValue());
					break;
				case BOOLEAN:
					System.out.println(buffer + so1.getBooleanValue());
					break;
				case STRING:
					System.out.println(buffer + so1.getStringValue());
					break;
				}
			}
		}
		else {
			/*
			 * Inputs for all builtins other than PRINT are treated as DOUBLE.
			 */
			BinaryOperator dop = (BinaryOperator) optr;
			double rval = dop.fn.execute(so1.getDoubleValue(), so2.getDoubleValue());
			
			sores = (ScalarObject) new DoubleObject(rval);
		}
		
		// 3) Put the result value into ProgramBlock
		ec.setScalarOutput(output.get_name(), sores);
	}
}
