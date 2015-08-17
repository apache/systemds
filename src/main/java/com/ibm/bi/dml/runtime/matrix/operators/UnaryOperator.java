/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.ValueFunction;

public class UnaryOperator extends Operator 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = 2441990876648978637L;

	public ValueFunction fn;
	public UnaryOperator(ValueFunction p)
	{
		fn = p;
		sparseSafe = false;
		
		if(fn instanceof Builtin)
		{
			Builtin f=(Builtin)fn;
			if(f.bFunc==Builtin.BuiltinFunctionCode.SIN || f.bFunc==Builtin.BuiltinFunctionCode.TAN 
					|| f.bFunc==Builtin.BuiltinFunctionCode.ROUND || f.bFunc==Builtin.BuiltinFunctionCode.ABS
					|| f.bFunc==Builtin.BuiltinFunctionCode.SQRT || f.bFunc==Builtin.BuiltinFunctionCode.SPROP
					|| f.bFunc==Builtin.BuiltinFunctionCode.LOG_NZ )
			{
				sparseSafe = true;
			}
		}
	}
}
