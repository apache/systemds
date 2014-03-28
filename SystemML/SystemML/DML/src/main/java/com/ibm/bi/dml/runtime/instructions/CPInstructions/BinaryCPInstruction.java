/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.functionobjects.And;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.Divide;
import com.ibm.bi.dml.runtime.functionobjects.Equals;
import com.ibm.bi.dml.runtime.functionobjects.GreaterThan;
import com.ibm.bi.dml.runtime.functionobjects.GreaterThanEquals;
import com.ibm.bi.dml.runtime.functionobjects.IntegerDivide;
import com.ibm.bi.dml.runtime.functionobjects.LessThan;
import com.ibm.bi.dml.runtime.functionobjects.LessThanEquals;
import com.ibm.bi.dml.runtime.functionobjects.Minus;
import com.ibm.bi.dml.runtime.functionobjects.Modulus;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Multiply2;
import com.ibm.bi.dml.runtime.functionobjects.NotEquals;
import com.ibm.bi.dml.runtime.functionobjects.Or;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.functionobjects.Power;
import com.ibm.bi.dml.runtime.functionobjects.Power2;
import com.ibm.bi.dml.runtime.functionobjects.Power2CMinus;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.LeftScalarOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.RightScalarOperator;
import com.ibm.bi.dml.runtime.matrix.operators.ScalarOperator;


public class BinaryCPInstruction extends ComputationCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public BinaryCPInstruction(Operator op, 
							 CPOperand in1, 
							 CPOperand in2, 
							 CPOperand out, 
						     String istr ){
		super(op, in1, in2, out);
		instString = istr;
	}

	public BinaryCPInstruction(Operator op, 
			 CPOperand in1, 
			 CPOperand in2, 
			 CPOperand in3, 
			 CPOperand out, 
		     String istr ){
		super(op, in1, in2, in3, out);
		instString = istr;
	}

	static String parseBinaryInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand out)
		throws DMLRuntimeException{
		
		InstructionUtils.checkNumFields ( instr, 3 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
		String opcode = parts[0];
		in1.split(parts[1]);
		in2.split(parts[2]);
		out.split(parts[3]);
		
		return opcode;
	}
	
	static String parseBinaryInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out)
	throws DMLRuntimeException{
	
		InstructionUtils.checkNumFields ( instr, 4 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
		String opcode = parts[0];
		in1.split(parts[1]);
		in2.split(parts[2]);
		in3.split(parts[3]);
		out.split(parts[4]);
		
		return opcode;
	}
	
	//what follows is all the binary operators that we currently allow
	
	//scalar-scalar or matrix-matrix operator 
	//is searched for by this function
	public static BinaryOperator getBinaryOperator(String opcode) throws DMLRuntimeException{
		if(opcode.equalsIgnoreCase("=="))
			return new BinaryOperator(Equals.getEqualsFnObject());
		else if(opcode.equalsIgnoreCase("!="))
			return new BinaryOperator(NotEquals.getNotEqualsFnObject());
		else if(opcode.equalsIgnoreCase("<"))
			return new BinaryOperator(LessThan.getLessThanFnObject());
		else if(opcode.equalsIgnoreCase(">"))
			return new BinaryOperator(GreaterThan.getGreaterThanFnObject());
		else if(opcode.equalsIgnoreCase("<="))
			return new BinaryOperator(LessThanEquals.getLessThanEqualsFnObject());
		else if(opcode.equalsIgnoreCase(">="))
			return new BinaryOperator(GreaterThanEquals.getGreaterThanEqualsFnObject());
		else if(opcode.equalsIgnoreCase("&&"))
			return new BinaryOperator(And.getAndFnObject());
		else if(opcode.equalsIgnoreCase("||"))
			return new BinaryOperator(Or.getOrFnObject());
		else if(opcode.equalsIgnoreCase("+"))
			return new BinaryOperator(Plus.getPlusFnObject());
		else if(opcode.equalsIgnoreCase("-"))
			return new BinaryOperator(Minus.getMinusFnObject());
		else if(opcode.equalsIgnoreCase("*"))
			return new BinaryOperator(Multiply.getMultiplyFnObject());
		else if ( opcode.equalsIgnoreCase("*2") ) 
			return new BinaryOperator(Multiply2.getMultiply2FnObject());
		else if(opcode.equalsIgnoreCase("/"))
			return new BinaryOperator(Divide.getDivideFnObject());
		else if(opcode.equalsIgnoreCase("%%"))
			return new BinaryOperator(Modulus.getModulusFnObject());
		else if(opcode.equalsIgnoreCase("%/%"))
			return new BinaryOperator(IntegerDivide.getIntegerDivideFnObject());
		else if(opcode.equalsIgnoreCase("^"))
			return new BinaryOperator(Power.getPowerFnObject());
		else if ( opcode.equalsIgnoreCase("^2") )
			return new BinaryOperator(Power2.getPower2FnObject());
		else if ( opcode.equalsIgnoreCase("max") ) 
			return new BinaryOperator(Builtin.getBuiltinFnObject("max"));
		else if ( opcode.equalsIgnoreCase("min") ) 
			return new BinaryOperator(Builtin.getBuiltinFnObject("min"));
		
		throw new DMLRuntimeException("Unknown binary opcode " + opcode);
	}
	
	//scalar-matrix operator is searched for by this function
	static ScalarOperator getScalarOperator(String opcode,
											boolean arg1IsScalar)
		throws DMLRuntimeException{
		double default_constant = 0;
		
		//commutative operators
		if ( opcode.equalsIgnoreCase("+") ){ 
			return new RightScalarOperator(Plus.getPlusFnObject(), default_constant); 
		}
		else if ( opcode.equalsIgnoreCase("*") ) {
			return new RightScalarOperator(Multiply.getMultiplyFnObject(), default_constant);
		} 
		//non-commutative operators but both scalar-matrix and matrix-scalar makes sense
		else if ( opcode.equalsIgnoreCase("-") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(Minus.getMinusFnObject(), default_constant);
			else return new RightScalarOperator(Minus.getMinusFnObject(), default_constant);
		} 
		else if ( opcode.equalsIgnoreCase("/") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(Divide.getDivideFnObject(), default_constant);
			else return new RightScalarOperator(Divide.getDivideFnObject(), default_constant);
		}  
		else if ( opcode.equalsIgnoreCase("%%") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(Modulus.getModulusFnObject(), default_constant);
			else return new RightScalarOperator(Modulus.getModulusFnObject(), default_constant);
		}
		else if ( opcode.equalsIgnoreCase("%/%") ) {
			if(arg1IsScalar)
				return new LeftScalarOperator(IntegerDivide.getIntegerDivideFnObject(), default_constant);
			else return new RightScalarOperator(IntegerDivide.getIntegerDivideFnObject(), default_constant);
		}
		//operations for which only matrix-scalar makes sense
		else if ( opcode.equalsIgnoreCase("^") ){
			if(arg1IsScalar)
				return new LeftScalarOperator(Power.getPowerFnObject(), default_constant);
			else return new RightScalarOperator(Power.getPowerFnObject(), default_constant);
		}
		else if ( opcode.equalsIgnoreCase("max") ) {
			return new RightScalarOperator(Builtin.getBuiltinFnObject("max"), default_constant);
		}
		else if ( opcode.equalsIgnoreCase("min") ) {
			return new RightScalarOperator(Builtin.getBuiltinFnObject("min"), default_constant);
		}
		else if ( opcode.equalsIgnoreCase("log") ){
			return new RightScalarOperator(Builtin.getBuiltinFnObject("log"), default_constant);
		}
		else if ( opcode.equalsIgnoreCase(">") ) {
			return new RightScalarOperator(GreaterThan.getGreaterThanFnObject(), default_constant);
		}
		else if ( opcode.equalsIgnoreCase(">=") ) {
			return new RightScalarOperator(GreaterThanEquals.getGreaterThanEqualsFnObject(), default_constant);
		}
		else if ( opcode.equalsIgnoreCase("<") ) {
			return new RightScalarOperator(LessThan.getLessThanFnObject(), default_constant);
		}
		else if ( opcode.equalsIgnoreCase("<=") ) {
			return new RightScalarOperator(LessThanEquals.getLessThanEqualsFnObject(), default_constant);
		}
		else if ( opcode.equalsIgnoreCase("==") ) {
			return new RightScalarOperator(Equals.getEqualsFnObject(), default_constant);
		}
		else if ( opcode.equalsIgnoreCase("!=") ) {
			return new RightScalarOperator(NotEquals.getNotEqualsFnObject(), default_constant);
		}
		
		//operation that only exist for performance purposes
		else if ( opcode.equalsIgnoreCase("*2") ) {
			return new RightScalarOperator(Multiply2.getMultiply2FnObject(), default_constant);
		} 
		else if ( opcode.equalsIgnoreCase("^2") ){
			return new RightScalarOperator(Power2.getPower2FnObject(), default_constant);
		}
		else if ( opcode.equalsIgnoreCase("^2c-") ){
			return new RightScalarOperator(Power2CMinus.getPower2CMFnObject(), default_constant);
		}
		
		throw new DMLRuntimeException("Unknown binary opcode " + opcode);
	}
}
