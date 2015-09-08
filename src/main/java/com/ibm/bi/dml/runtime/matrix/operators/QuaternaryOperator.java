/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */


package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.lops.WeightedDivMM.WDivMMType;
import com.ibm.bi.dml.lops.WeightedSigmoid.WSigmoidType;
import com.ibm.bi.dml.lops.WeightedSquaredLoss.WeightsType;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.FunctionObject;

public class QuaternaryOperator extends Operator 
{

	private static final long serialVersionUID = -1642908613016116069L;

	public WeightsType wtype1 = null;
	public WSigmoidType wtype2 = null;
	public WDivMMType wtype3 = null;
	public FunctionObject fn;
	
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
}
