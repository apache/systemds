/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.functionobjects.GreaterThan;
import com.ibm.bi.dml.runtime.functionobjects.GreaterThanEquals;
import com.ibm.bi.dml.runtime.functionobjects.LessThan;
import com.ibm.bi.dml.runtime.functionobjects.LessThanEquals;
import com.ibm.bi.dml.runtime.functionobjects.Power;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;


public class LeftScalarOperator extends ScalarOperator 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 2360577666575746424L;
	
	public LeftScalarOperator(ValueFunction p, double cst) {
		super(p, cst);
		
		//disable sparse-safe for c^M because 1^0=1
		if( fn instanceof Power )
			sparseSafe = false;
	}

	@Override
	public double executeScalar(double in) throws DMLRuntimeException {
		return fn.execute(_constant, in);
	}
	
	@Override
	public void setConstant(double cst) 
	{
		super.setConstant(cst);
		
		//disable sparse-safe for c^M because 1^0=1
		if( fn instanceof Power )
			sparseSafe = false;
		
		//enable conditionally sparse safe operations
		if(    (fn instanceof GreaterThan && _constant<=0)
			|| (fn instanceof GreaterThanEquals && _constant<0)
			|| (fn instanceof LessThan && _constant>=0)
			|| (fn instanceof LessThanEquals && _constant>0))
		{
			sparseSafe = true;
		}
	}
}
