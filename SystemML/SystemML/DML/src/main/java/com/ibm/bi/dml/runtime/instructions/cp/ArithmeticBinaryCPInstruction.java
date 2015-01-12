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
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public abstract class ArithmeticBinaryCPInstruction extends BinaryCPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	public ArithmeticBinaryCPInstruction(Operator op, 
								   CPOperand in1, 
								   CPOperand in2, 
								   CPOperand out, 
								   String istr )
	{
		super(op, in1, in2, out, istr);
		cptype = CPINSTRUCTION_TYPE.ArithmeticBinary;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException, DMLUnsupportedOperationException {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = parseBinaryInstruction(str, in1, in2, out);
		
		// Arithmetic operations must be performed on DOUBLE or INT
		ValueType vt1 = in1.get_valueType();
		DataType dt1 = in1.get_dataType();
		ValueType vt2 = in2.get_valueType();
		DataType dt2 = in2.get_dataType();
		ValueType vt3 = out.get_valueType();
		DataType dt3 = out.get_dataType();
		
		//prithvi TODO
		//make sure these checks belong here
		//if either input is a matrix, then output
		//has to be a matrix
		if((dt1 == DataType.MATRIX 
			|| dt2 == DataType.MATRIX) 
		   && dt3 != DataType.MATRIX)
			throw new DMLRuntimeException("Element-wise matrix operations between variables "
										  + in1.get_name()
										  + " and "
										  + in2.get_name()
										  + " must produce a matrix, which "
										  + out.get_name()
										  + "is not");
		
		Operator operator = 
			(dt1 != dt2) ?
					getScalarOperator(opcode, (dt1 == DataType.SCALAR))
					: getBinaryOperator(opcode);
		
		if ( opcode.equalsIgnoreCase("+") && dt1 == DataType.SCALAR && dt2 == DataType.SCALAR) {
			return new ScalarScalarArithmeticCPInstruction(operator, 
														   in1, 
														   in2, 
														   out, 
														   str);
		} else if(dt1 == DataType.SCALAR && dt2 == DataType.SCALAR){
			if ( (vt1 != ValueType.DOUBLE && vt1 != ValueType.INT)
					|| (vt2 != ValueType.DOUBLE && vt2 != ValueType.INT)
					|| (vt3 != ValueType.DOUBLE && vt3 != ValueType.INT) )
				throw new DMLRuntimeException("Unexpected ValueType in ArithmeticInstruction.");
			
			//haven't we already checked for this above -- prithvi
			if ( vt1 != ValueType.DOUBLE && vt1 != ValueType.INT ) {
				throw new DMLRuntimeException("Unexpected ValueType (" + vt1 + ") in ArithmeticInstruction: " + str);
			}
			
			return new ScalarScalarArithmeticCPInstruction(operator, in1, in2, out, str);
		
		} else if (dt1 == DataType.MATRIX || dt2 == DataType.MATRIX){
			if(vt1 == ValueType.STRING 
			   || vt2 == ValueType.STRING 
			   || vt3 == ValueType.STRING)
				throw new DMLUnsupportedOperationException("We do not support element-wise string operations on matrices "
												  + in1.get_name()
												  + ", "
												  + in2.get_name()
												  + " and "
												  + out.get_name());
				
			if(dt1 == DataType.MATRIX && dt2 == DataType.MATRIX)
				return new MatrixMatrixArithmeticCPInstruction(operator, in1, in2, out, str);
			else
				return new ScalarMatrixArithmeticCPInstruction(operator, in1, in2, out, str);	
		}
		
		return null;
	}
}
