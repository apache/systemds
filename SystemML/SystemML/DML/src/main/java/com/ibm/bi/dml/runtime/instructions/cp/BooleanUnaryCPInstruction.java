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
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;


public class BooleanUnaryCPInstruction extends UnaryCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public BooleanUnaryCPInstruction(Operator op,
									 CPOperand in,
									 CPOperand out,
									 String opcode,
									 String instr){
		super(op, in, out, opcode, instr);
		_cptype = CPINSTRUCTION_TYPE.BooleanUnary;
	}

	public static Instruction parseInstruction (String str) throws DMLRuntimeException {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = parseUnaryInstruction(str, in, out);
		
		// Boolean operations must be performed on BOOLEAN
		ValueType vt1, vt2;
		vt1 = vt2 = null;
		vt1 = in.get_valueType();
		vt2 = out.get_valueType();
		if ( vt1 != ValueType.BOOLEAN || vt2 != ValueType.BOOLEAN )
			throw new DMLRuntimeException("Unexpected ValueType in ArithmeticInstruction.");
		
		// Determine appropriate Function Object based on opcode	
		return new BooleanUnaryCPInstruction(getSimpleUnaryOperator(opcode), in, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException {
		// 1) Obtain data objects associated with inputs 
		ScalarObject so = ec.getScalarInput(input1.get_name(), input1.get_valueType(), input1.isLiteral());
		ScalarObject sores = null;
		
		// 2) Compute the result value & make an appropriate data object 
		SimpleOperator dop = (SimpleOperator) _optr;
		boolean rval;
		rval = dop.fn.execute(so.getBooleanValue());
		
		sores = (ScalarObject) new BooleanObject(rval);
		
		// 3) Put the result value into ProgramBlock
		ec.setScalarOutput(output.get_name(), sores);
	}
}
