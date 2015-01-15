/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLScriptException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;


public class ScalarBuiltinCPInstruction extends BuiltinUnaryCPInstruction
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public ScalarBuiltinCPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String instr)
	{
		super(op, in, out, 1, opcode, instr);
	}
	
	@Override 
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException 
	{	
		String opcode = getOpcode();
		SimpleOperator dop = (SimpleOperator) _optr;
		ScalarObject sores = null;
		ScalarObject so = null;
		
		//get the scalar input 
		so = ec.getScalarInput( input1.get_name(), input1.get_valueType(), input1.isLiteral() );
			
		//core execution
		if ( opcode.equalsIgnoreCase("print") ) {
			String outString = so.getStringValue();
			
			// print to stdout only when suppress flag in DMLScript is not set.
			// The flag will be set, for example, when SystemML is invoked in fenced mode from Jaql.
			if (!DMLScript.suppressPrint2Stdout())
				System.out.println(outString);
			
			// String that is printed on stdout will be inserted into symbol table (dummy, not necessary!) 
			sores = new StringObject(outString);
		}
		else if ( opcode.equalsIgnoreCase("stop") ) {
			String msg = so.getStringValue();
			throw new DMLScriptException(msg);
		}
		else {
			//Inputs for all builtins other than PRINT are treated as DOUBLE.
			double rval;
			rval = dop.fn.execute(so.getDoubleValue());
			sores = (ScalarObject) new DoubleObject(rval);
		}
		
		ec.setScalarOutput(output.get_name(), sores);
	}

}
