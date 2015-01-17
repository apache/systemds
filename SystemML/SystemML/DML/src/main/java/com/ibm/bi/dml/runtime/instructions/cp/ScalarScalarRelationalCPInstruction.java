/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class ScalarScalarRelationalCPInstruction extends RelationalBinaryCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public ScalarScalarRelationalCPInstruction(Operator op, 
											   CPOperand in1, 
											   CPOperand in2,
											   CPOperand out, 
											   String opcode,
											   String istr){
		super(op, in1, in2, out, opcode, istr);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException{
		ScalarObject so1 = ec.getScalarInput(input1.getName(), input1.getValueType(), input1.isLiteral());
		ScalarObject so2 = ec.getScalarInput(input2.getName(), input2.getValueType(), input2.isLiteral() );
		ScalarObject sores = null;
		
		BinaryOperator dop = (BinaryOperator) _optr;
		
		/*if ( input1.getValueType() == ValueType.INT && input2.getValueType() == ValueType.INT ) {
			boolean rval = dop.fn.compare ( so1.getIntValue(), so2.getIntValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		else if ( input1.getValueType() == ValueType.DOUBLE && input2.getValueType() == ValueType.DOUBLE ) {
			boolean rval = dop.fn.compare ( so1.getDoubleValue(), so2.getDoubleValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		else if ( input1.getValueType() == ValueType.INT && input2.getValueType() == ValueType.DOUBLE ) {
			boolean rval = dop.fn.compare ( so1.getIntValue(), so2.getDoubleValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		else if ( input1.getValueType() == ValueType.DOUBLE && input2.getValueType() == ValueType.INT ) {
			boolean rval = dop.fn.compare ( so1.getDoubleValue(), so2.getIntValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		else if ( input1.getValueType() == ValueType.BOOLEAN && input2.getValueType() == ValueType.BOOLEAN ) {
			boolean rval = dop.fn.compare ( so1.getBooleanValue(), so2.getBooleanValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		else throw new DMLRuntimeException("compare(): Invalid combination of value types.");
		*/
		if ( so1 instanceof IntObject && so2 instanceof IntObject ) {
			boolean rval = dop.fn.compare ( so1.getLongValue(), so2.getLongValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		else if ( so1 instanceof DoubleObject && so2 instanceof DoubleObject ) {
			boolean rval = dop.fn.compare ( so1.getDoubleValue(), so2.getDoubleValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		else if ( so1 instanceof IntObject && so2 instanceof DoubleObject) {
			boolean rval = dop.fn.compare ( so1.getLongValue(), so2.getDoubleValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		else if ( so1 instanceof DoubleObject && so2 instanceof IntObject ) {
			boolean rval = dop.fn.compare ( so1.getDoubleValue(), so2.getLongValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		else if ( so1 instanceof BooleanObject && so2 instanceof BooleanObject ) {
			boolean rval = dop.fn.compare ( so1.getBooleanValue(), so2.getBooleanValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		else if ( so1 instanceof StringObject && so2 instanceof StringObject ) {
			boolean rval = dop.fn.compare ( so1.getStringValue(), so2.getStringValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		else throw new DMLRuntimeException("compare(): Invalid combination of value types.");
		
		ec.setScalarOutput(output.getName(), sores);
	}
}
