/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.operators;

import java.io.Serializable;

import com.ibm.bi.dml.hops.Hop.OpOp2;
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
import com.ibm.bi.dml.runtime.functionobjects.MinusNz;
import com.ibm.bi.dml.runtime.functionobjects.Modulus;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.NotEquals;
import com.ibm.bi.dml.runtime.functionobjects.Or;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.functionobjects.Power;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;
import com.ibm.bi.dml.runtime.functionobjects.Builtin.BuiltinFunctionCode;

public class BinaryOperator  extends Operator implements Serializable
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = -2547950181558989209L;

	public ValueFunction fn;
	
	public BinaryOperator(ValueFunction p)
	{
		fn=p;
		//as long as (0 op 0)=0, then op is sparseSafe
		if(fn instanceof Plus || fn instanceof Multiply || fn instanceof Minus 
				|| fn instanceof And || fn instanceof Or)
			sparseSafe=true;
		else
			sparseSafe=false;
	}
	
	/**
	 * Method for getting the hop binary operator type for a given function object.
	 * This is used in order to use a common code path for consistency between 
	 * compiler and runtime.
	 * 
	 * @return
	 */
	public OpOp2 getBinaryOperatorOpOp2()
	{
		if( fn instanceof Plus )				return OpOp2.PLUS;
		else if( fn instanceof Minus )			return OpOp2.MINUS;
		else if( fn instanceof Multiply )		return OpOp2.MULT;
		else if( fn instanceof Divide )			return OpOp2.DIV;
		else if( fn instanceof Modulus )		return OpOp2.MODULUS;
		else if( fn instanceof IntegerDivide )	return OpOp2.INTDIV;
		else if( fn instanceof LessThan )		return OpOp2.LESS;
		else if( fn instanceof LessThanEquals )	return OpOp2.LESSEQUAL;
		else if( fn instanceof GreaterThan )	return OpOp2.GREATER;
		else if( fn instanceof GreaterThanEquals )	return OpOp2.GREATEREQUAL;
		else if( fn instanceof Equals )			return OpOp2.EQUAL;
		else if( fn instanceof NotEquals )		return OpOp2.NOTEQUAL;
		else if( fn instanceof And )			return OpOp2.AND;
		else if( fn instanceof Or )				return OpOp2.OR;
		else if( fn instanceof Power )			return OpOp2.POW;
		else if( fn instanceof MinusNz )		return OpOp2.MINUS_NZ;
		else if( fn instanceof Builtin ) {
			BuiltinFunctionCode bfc = ((Builtin) fn).getBuiltinFunctionCode();
			if( bfc == BuiltinFunctionCode.MIN ) 		return OpOp2.MIN;
			else if( bfc == BuiltinFunctionCode.MAX ) 	return OpOp2.MAX;
			else if( bfc == BuiltinFunctionCode.LOG ) 	return OpOp2.LOG;
			else if( bfc == BuiltinFunctionCode.LOG_NZ ) return OpOp2.LOG_NZ;
		}
		
		//non-supported ops (not required for sparsity estimates):
		//PRINT, CONCAT, QUANTILE, INTERQUANTILE, IQM, 
		//CENTRALMOMENT, COVARIANCE, APPEND, SEQINCR, SOLVE, MEDIAN,
			
		return OpOp2.INVALID;
	}
}
