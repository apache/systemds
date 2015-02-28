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
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class BooleanBinaryCPInstruction extends BinaryCPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public BooleanBinaryCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr )
	{
		super(op, in1, in2, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.BooleanBinary;
	}
	
	public static Instruction parseInstruction (String str) throws DMLRuntimeException {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = parseBinaryInstruction(str, in1, in2, out);
		
		// Boolean operations must be performed on BOOLEAN
		ValueType vt1 = in1.getValueType();
		ValueType vt2 = in2.getValueType();
		ValueType vt3 = out.getValueType();
		if ( vt1 != ValueType.BOOLEAN || vt3 != ValueType.BOOLEAN 
				|| (vt2 != null && vt2 != ValueType.BOOLEAN) )
			throw new DMLRuntimeException("Unexpected ValueType in ArithmeticInstruction.");
		
		// Determine appropriate Function Object based on opcode	
		return new BooleanBinaryCPInstruction(getBinaryOperator(opcode), in1, in2, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException 
	{
		ScalarObject so1 = ec.getScalarInput( input1.getName(), input1.getValueType(), input1.isLiteral() );
		ScalarObject so2 = ec.getScalarInput( input2.getName(), input2.getValueType(), input2.isLiteral() );
		
		BinaryOperator dop = (BinaryOperator) _optr;
		boolean rval = dop.fn.execute(so1.getBooleanValue(), so2.getBooleanValue());
		ScalarObject sores = (ScalarObject) new BooleanObject(rval);
		
		ec.setScalarOutput(output.getName(), sores);
	}
}
