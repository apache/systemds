/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.functionobjects.And;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.Builtin.BuiltinFunctionCode;
import com.ibm.bi.dml.runtime.functionobjects.Equals;
import com.ibm.bi.dml.runtime.functionobjects.GreaterThan;
import com.ibm.bi.dml.runtime.functionobjects.LessThan;
import com.ibm.bi.dml.runtime.functionobjects.Minus;
import com.ibm.bi.dml.runtime.functionobjects.MinusNz;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Multiply2;
import com.ibm.bi.dml.runtime.functionobjects.NotEquals;
import com.ibm.bi.dml.runtime.functionobjects.Power;
import com.ibm.bi.dml.runtime.functionobjects.Power2;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;


public class ScalarOperator  extends Operator 
{
	private static final long serialVersionUID = 4547253761093455869L;

	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public ValueFunction fn;
	protected double _constant;
	
	public ScalarOperator(ValueFunction p, double cst)
	{
		fn = p;
		_constant = cst;
		
		//as long as (0 op v)=0, then op is sparsesafe
		//note: additional functionobjects might qualify according to constant
		if(   fn instanceof Multiply || fn instanceof Multiply2 
		   || fn instanceof Power || fn instanceof Power2 
		   || fn instanceof And || fn instanceof MinusNz
		   || (fn instanceof Builtin && ((Builtin)fn).getBuiltinFunctionCode()==BuiltinFunctionCode.LOG_NZ)) 
		{
			sparseSafe=true;
		}
		else
		{
			sparseSafe=false;
		}
	}
	
	public double getConstant()
	{
		return _constant;
	}
	
	public void setConstant(double cst) 
	{
		//set constant
		_constant = cst;
		
		//revisit sparse safe decision according to known constant
		//note: there would be even more potential if we take left/right op into account
		if(    fn instanceof Multiply || fn instanceof Multiply2 
			|| fn instanceof Power || fn instanceof Power2 
			|| fn instanceof And || fn instanceof MinusNz
			|| fn instanceof Builtin && ((Builtin)fn).getBuiltinFunctionCode()==BuiltinFunctionCode.LOG_NZ
			|| (fn instanceof GreaterThan && _constant==0) 
			|| (fn instanceof LessThan && _constant==0)
			|| (fn instanceof NotEquals && _constant==0)
			|| (fn instanceof Equals && _constant!=0)
			|| (fn instanceof Minus && _constant==0))
		{
			sparseSafe = true;
		}
		else
		{
			sparseSafe = false;
		}
	}
	
	public double executeScalar(double in) throws DMLRuntimeException {
		throw new DMLRuntimeException("executeScalar(): can not be invoked from base class.");
	}
}
