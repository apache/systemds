/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class ScalarScalarBuiltinCPInstruction extends BuiltinBinaryCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public ScalarScalarBuiltinCPInstruction(Operator op,
			  								CPOperand in1,
			  								CPOperand in2,
			  								CPOperand out,
			  								String opcode,
			  								String instr){
		super(op, in1, in2, out, 2, opcode, instr);
	}

	@Override 
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException {
		
		String opcode = getOpcode();
		ScalarObject sores = null;
		
		ScalarObject so1 = ec.getScalarInput( input1.getName(), input1.getValueType(), input1.isLiteral() );
		ScalarObject so2 = ec.getScalarInput(input2.getName(), input2.getValueType(), input2.isLiteral() );
		
		if ( opcode.equalsIgnoreCase("print") ) {
			String buffer = "";
			if (input2.getValueType() != ValueType.STRING)
				throw new DMLRuntimeException("wrong value type in print");
			buffer = so2.getStringValue() + " ";
			
			if ( !DMLScript.suppressPrint2Stdout()) {
				switch (input1.getValueType()) {
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
					default:
						//do nothing
				}
			}
		}
		else {
			/*
			 * Inputs for all builtins other than PRINT are treated as DOUBLE.
			 */
			BinaryOperator dop = (BinaryOperator) _optr;
			double rval = dop.fn.execute(so1.getDoubleValue(), so2.getDoubleValue());
			
			sores = (ScalarObject) new DoubleObject(rval);
		}
		
		// 3) Put the result value into ProgramBlock
		ec.setScalarOutput(output.getName(), sores);
	}
}
