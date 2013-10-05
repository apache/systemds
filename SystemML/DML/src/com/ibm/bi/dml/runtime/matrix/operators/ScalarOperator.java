/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.functionobjects.And;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;


public class ScalarOperator  extends Operator 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public ValueFunction fn;
	public double constant;
	public ScalarOperator(ValueFunction p, double cst)
	{
		fn=p;
		constant=cst;
		//as long as (0 op v)=0, then op is sparsesafe
		if(fn instanceof Multiply || fn instanceof And)
			sparseSafe=true;
		else
			sparseSafe=false;
	}
	public void setConstant(double cst) {
		constant = cst;
	}
	public double executeScalar(double in) throws DMLRuntimeException {
		throw new DMLRuntimeException("executeScalar(): can not be invoked from base class.");
	}
}
