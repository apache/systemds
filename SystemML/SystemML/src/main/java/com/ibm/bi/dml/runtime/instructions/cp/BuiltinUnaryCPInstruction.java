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
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;
import com.ibm.bi.dml.runtime.matrix.operators.UnaryOperator;


public abstract class BuiltinUnaryCPInstruction extends UnaryCPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	int arity;
	
	public BuiltinUnaryCPInstruction(Operator op, CPOperand in, CPOperand out, int _arity, String opcode, String istr )
	{
		super(op, in, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.BuiltinUnary;
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
		
		if( parts.length==4 ) //print or stop
		{
			opcode = parts[0];
			in.split(parts[1]);
			out.split(parts[2]);
			func = Builtin.getBuiltinFnObject(opcode);
			
			return new ScalarBuiltinCPInstruction(new SimpleOperator(func), in, out, opcode, str);
		}
		else //2+1, general case
		{
			opcode = parseUnaryInstruction(str, in, out);
			func = Builtin.getBuiltinFnObject(opcode);
			
			if(in.getDataType() == DataType.SCALAR)
				return new ScalarBuiltinCPInstruction(new SimpleOperator(func), in, out, opcode, str);
			else if(in.getDataType() == DataType.MATRIX)
				return new MatrixBuiltinCPInstruction(new UnaryOperator(func), in, out, opcode, str);
		}
		
		return null;
	}
}
