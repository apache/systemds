/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;
import com.ibm.bi.dml.runtime.matrix.operators.UnaryOperator;


public class BuiltinUnaryCPInstruction extends UnaryCPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	int arity;
	
	public BuiltinUnaryCPInstruction(Operator op, CPOperand in, CPOperand out, int _arity, String istr )
	{
		super(op, in, out, istr);
		cptype = CPINSTRUCTION_TYPE.BuiltinUnary;
		instString = istr;
		arity = _arity;
	}

	public int getArity() {
		return arity;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = null;
		ValueFunction func = null;
		
		if( parts.length==4 ) //print, print2
		{
			opcode = parts[0];
			boolean literal = Boolean.parseBoolean(parts[3]);
			in.split(parts[1]);
			out.split(parts[2]);
			func = Builtin.getBuiltinFnObject(opcode);
			
			return new ScalarBuiltinCPInstruction(new SimpleOperator(func), in, out, str, literal);
		}
		else //2+1, general case
		{
			opcode = parseUnaryInstruction(str, in, out);
			func = Builtin.getBuiltinFnObject(opcode);
			
			if(in.get_dataType() == DataType.SCALAR)
				return new ScalarBuiltinCPInstruction(new SimpleOperator(func), in, out, str);
			else if(in.get_dataType() == DataType.MATRIX)
				return new MatrixBuiltinCPInstruction(new UnaryOperator(func), in, out, str);
		}
		
		/*
	    int _arity = parts.length - 2;
		boolean b = ((Builtin) func).checkArity(arity);
		if ( !b ) {
			throw new DMLRuntimeException("Invalid number of inputs to builtin function: " 
										  + ((Builtin) func).bFunc);
		}
		*/
		
		// TODO: VALUE TYPE CHECKING
		/*
		ValueType vt1, vt2, vt3;
		vt1 = vt2 = vt3 = null;
		vt1 = in1.get_valueType();
		if ( in2 != null )
			vt2 = in2.get_valueType();
		vt3 = out.get_valueType();
		if ( vt1 != ValueType.BOOLEAN || vt3 != ValueType.BOOLEAN 
				|| (vt2 != null && vt2 != ValueType.BOOLEAN) )
			throw new DMLRuntimeException("Unexpected ValueType in ArithmeticInstruction.");
		*/
		
		return null;
	}
}
