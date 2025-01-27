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


package org.apache.sysds.runtime.matrix.operators;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.lops.WeightedCrossEntropy.WCeMMType;
import org.apache.sysds.lops.WeightedDivMM.WDivMMType;
import org.apache.sysds.lops.WeightedSigmoid.WSigmoidType;
import org.apache.sysds.lops.WeightedSquaredLoss.WeightsType;
import org.apache.sysds.lops.WeightedUnaryMM.WUMMType;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Multiply2;
import org.apache.sysds.runtime.functionobjects.Power2;
import org.apache.sysds.runtime.functionobjects.ValueFunction;

public class QuaternaryOperator extends Operator 
{
	private static final long serialVersionUID = -1642908613016116069L;

	public final WeightsType wtype1;
	public final WSigmoidType wtype2;
	public final WDivMMType wtype3;
	public final WCeMMType wtype4;
	public final WUMMType wtype5;
	
	public final ValueFunction fn;
	private final double eps;

	private QuaternaryOperator( WeightsType wt1, WSigmoidType wt2, WDivMMType wt3, WCeMMType wt4, WUMMType wt5, ValueFunction fn, double eps ) {
		wtype1 = wt1;
		wtype2 = wt2;
		wtype3 = wt3;
		wtype4 = wt4;
		wtype5 = wt5;
		this.fn = fn;
		this.eps = eps;
	}
	
	/**
	 * wsloss
	 * 
	 * @param wt Weights type
	 */
	public QuaternaryOperator( WeightsType wt ) {
		this(wt, null, null, null, null, null, 0);
	}
	
	/**
	 * wsigmoid 
	 * 
	 * @param wt WSigmoid type
	 */
	public QuaternaryOperator( WSigmoidType wt ) {
		this(null, wt, null, null, null, Builtin.getBuiltinFnObject("sigmoid"), 0);
	}
	
	/**
	 * wdivmm
	 * 
	 * @param wt WDivMM type
	 */
	public QuaternaryOperator( WDivMMType wt ) {
		this(null, null, wt, null, null, null, 0);
	}
	
	/**
	 * wdivmm w/epsilon
	 * 
	 * @param wt WDivMM type
	 * @param epsilon the epsilon value
	 */
	public QuaternaryOperator( WDivMMType wt, double epsilon) {
		this(null, null, wt, null, null, null, epsilon);
	}
	
	/**
	 * wcemm
	 * 
	 * @param wt WCeMM type
	 */
	public QuaternaryOperator( WCeMMType wt ) {
		this(null, null, null, wt, null, null, 0);
	}
	
	/**
	 * wcemm w/epsilon
	 * 
	 * @param wt WCeMM type
	 * @param epsilon the epsilon value
	 */
	public QuaternaryOperator( WCeMMType wt, double epsilon) {
		this(null, null, null, wt, null, null, epsilon);
	}
	
	/**
	 * wumm
	 * 
	 * @param wt WUMM type
	 * @param op operator type
	 */
	public QuaternaryOperator( WUMMType wt, String op ) {
		this(null, null, null, null, wt, 
			op.equals(Opcodes.POW2.toString()) ? Power2.getPower2FnObject() :
			op.equals(Opcodes.MULT2.toString()) ? Multiply2.getMultiply2FnObject() :
			Builtin.getBuiltinFnObject(op), 0);
	}

	public boolean hasFourInputs() {
		return (wtype1 != null && wtype1.hasFourInputs())
			|| (wtype3 != null && wtype3.hasFourInputs())
			|| (wtype4 != null && wtype4.hasFourInputs());
	}
	
	/**
	 * Obtain epsilon value
	 * 
	 * @return epsilon
	 */
	public double getScalar() {
		return eps;
	}
}
