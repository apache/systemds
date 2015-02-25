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
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public abstract class RelationalBinaryCPInstruction extends BinaryCPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public RelationalBinaryCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr )
	{
		super(op, in1, in2, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.RelationalBinary;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		InstructionUtils.checkNumFields (str, 3);
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = parseBinaryInstruction(str, in1, in2, out);
		
		// TODO: Relational operations need not have value type checking
		ValueType vt1 = in1.getValueType();
		DataType dt1 = in1.getDataType();
		ValueType vt2 = in2.getValueType();
		DataType dt2 = in2.getDataType();
		DataType dt3 = out.getDataType();
		
		//if ( vt3 != ValueType.BOOLEAN )
		//	throw new DMLRuntimeException("Unexpected ValueType in RelationalCPInstruction: " + str);
		
		if ( vt1 == ValueType.BOOLEAN && !opcode.equalsIgnoreCase("==") && !opcode.equalsIgnoreCase("!=") ) 
			throw new DMLRuntimeException("Operation " + opcode + " can not be applied on boolean values "
					 					  + "(Instruction = " + str + ").");
		
		//prithvi TODO
		//make sure these checks belong here
		//if either input is a matrix, then output
		//has to be a matrix
		if((dt1 == DataType.MATRIX 
			|| dt2 == DataType.MATRIX) 
		   && dt3 != DataType.MATRIX)
			throw new DMLRuntimeException("Element-wise matrix operations between variables "
										  + in1.getName()
										  + " and "
										  + in2.getName()
										  + " must produce a matrix, which "
										  + out.getName()
										  + " is not");
		
		Operator operator = 
			(dt1 != dt2) ?
					getScalarOperator(opcode, (dt1 == DataType.SCALAR))
					: getBinaryOperator(opcode);
		
		//for scalar relational operations we only allow boolean operands
		//or when both operands are numeric (int or double)
		if(dt1 == DataType.SCALAR && dt2 == DataType.SCALAR){
			if (!(  (vt1 == ValueType.BOOLEAN && vt2 == ValueType.BOOLEAN)
				  ||(vt1 == ValueType.STRING && vt2 == ValueType.STRING)
				  ||( (vt1 == ValueType.DOUBLE || vt1 == ValueType.INT) && (vt2 == ValueType.DOUBLE || vt2 == ValueType.INT))))
			{
				throw new DMLRuntimeException("unexpected value-type in "
											  + "Relational Binary Instruction "
											  + "involving scalar operands.");
			}
			return new ScalarScalarRelationalCPInstruction(operator, in1, in2, out, opcode, str);
		
		}else if (dt1 == DataType.MATRIX || dt2 == DataType.MATRIX){
			if(dt1 == DataType.MATRIX && dt2 == DataType.MATRIX)
				return new MatrixMatrixRelationalCPInstruction(operator, in1, in2, out, opcode, str);
			else
				return new ScalarMatrixRelationalCPInstruction(operator, in1, in2, out, opcode, str);
		}
		
		return null;
	}
}
