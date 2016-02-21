/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */


package org.apache.sysml.runtime.matrix.operators;

import org.apache.sysml.lops.WeightedDivMM.WDivMMType;
import org.apache.sysml.lops.WeightedSigmoid.WSigmoidType;
import org.apache.sysml.lops.WeightedSquaredLoss.WeightsType;
import org.apache.sysml.lops.WeightedCrossEntropy.WCeMMType;
import org.apache.sysml.lops.WeightedUnaryMM.WUMMType;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.Multiply2;
import org.apache.sysml.runtime.functionobjects.Power2;
import org.apache.sysml.runtime.functionobjects.ValueFunction;

public class QuaternaryOperator extends Operator 
{

	private static final long serialVersionUID = -1642908613016116069L;

	public WeightsType wtype1 = null;
	public WSigmoidType wtype2 = null;
	public WDivMMType wtype3 = null;
	public WCeMMType wtype4 = null;
	public WUMMType wtype5 = null;
	
	public ValueFunction fn;
	
	private double eps = 0;

	/**
	 * wsloss
	 * 
	 * @param wt
	 */
	public QuaternaryOperator( WeightsType wt ) {
		wtype1 = wt;
	}
	
	/**
	 * wsigmoid 
	 * 
	 * @param wt
	 */
	public QuaternaryOperator( WSigmoidType wt ) {
		wtype2 = wt;
		fn = Builtin.getBuiltinFnObject("sigmoid");
	}
	
	/**
	 * wdivmm
	 * 
	 * @param wt
	 */
	public QuaternaryOperator( WDivMMType wt ) {
		wtype3 = wt;
	}
	
	/**
	 * wdivmm w/epsilon
	 * 
	 * @param wt
	 */
	public QuaternaryOperator( WDivMMType wt, double epsilon) {
		wtype3 = wt;
		eps = epsilon;
	}
	
	/**
	 * wcemm
	 * 
	 * @param wt
	 */
	public QuaternaryOperator( WCeMMType wt ) {
		wtype4 = wt;
	}
	
	/**
	 * wumm
	 * 
	 * @param wt
	 * @param op
	 */
	public QuaternaryOperator( WUMMType wt, String op ) {
		wtype5 = wt;
		
		if( op.equals("^2") )
			fn = Power2.getPower2FnObject();
		else if( op.equals("*2") )
			fn = Multiply2.getMultiply2FnObject();
		else
			fn = Builtin.getBuiltinFnObject(op);
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean hasFourInputs() {
		return (wtype1 != null && wtype1.hasFourInputs())
			|| (wtype3 != null && wtype3.hasFourInputs());
	}
	
	/**
	 * 
	 * @return
	 */
	public double hasScalar() {
		return eps;
	}

}
