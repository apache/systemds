/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.lops.WeightedSigmoid.WSigmoidType;
import com.ibm.bi.dml.lops.WeightedSquaredLoss.WeightsType;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.FunctionObject;

public class QuaternaryOperator extends Operator 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = -1642908613016116069L;

	public WeightsType wtype1 = null;
	public WSigmoidType wtype2 = null;
	public FunctionObject fn;
	
	public QuaternaryOperator( WeightsType wt ) {
		wtype1 = wt;
	}
	
	public QuaternaryOperator( WSigmoidType wt ) {
		wtype2 = wt;
		fn = Builtin.getBuiltinFnObject("sigmoid");
	}
}
