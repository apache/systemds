/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.Divide;
import com.ibm.bi.dml.runtime.functionobjects.Power;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class ScalarScalarArithmeticCPInstruction extends ArithmeticBinaryCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public ScalarScalarArithmeticCPInstruction(Operator op, 
								   CPOperand in1, 
								   CPOperand in2,
								   CPOperand out, 
								   String istr){
		super(op, in1, in2, out, istr);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException{
		// 1) Obtain data objects associated with inputs 
		ScalarObject so1 = ec.getScalarInput(input1.get_name(), input1.get_valueType(), input1.isLiteral());
		ScalarObject so2 = ec.getScalarInput(input2.get_name(), input2.get_valueType(), input2.isLiteral() );
		ScalarObject sores = null;
		
		
		// 2) Compute the result value & make an appropriate data object 
		BinaryOperator dop = (BinaryOperator) optr;
		
		if ( input1.get_valueType() == ValueType.STRING 
			 || input2.get_valueType() == ValueType.STRING ) {
			String rval = dop.fn.execute(so1.getStringValue(), so2.getStringValue());
			sores = (ScalarObject) new StringObject(rval);
		}
		else if ( so1 instanceof IntObject && so2 instanceof IntObject ) {
			if ( dop.fn instanceof Divide || dop.fn instanceof Power ) {
				// If both inputs are of type INT then output must be an INT if operation is not divide or power
				double rval = dop.fn.execute ( so1.getIntValue(), so2.getIntValue() );
				sores = (ScalarObject) new DoubleObject(rval);
			}
			else {
				// If both inputs are of type INT then output must be an INT if operation is not divide or power
				int rval = (int) dop.fn.execute ( so1.getIntValue(), so2.getIntValue() );
				sores = (ScalarObject) new IntObject(rval);
			}
		}
		
		else {
			// If either of the input is of type DOUBLE then output is a DOUBLE
			double rval = dop.fn.execute ( so1.getDoubleValue(), so2.getDoubleValue() );
			sores = (ScalarObject) new DoubleObject(rval); 
		}
		
		// 3) Put the result value into ProgramBlock
		ec.setScalarOutput(output.get_name(), sores);
	}
}
