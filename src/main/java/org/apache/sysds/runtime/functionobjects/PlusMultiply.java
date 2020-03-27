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

package org.apache.sysds.runtime.functionobjects;

import java.io.Serializable;

import org.apache.sysds.runtime.functionobjects.TernaryValueFunction.ValueFunctionWithConstant;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;

public class PlusMultiply extends TernaryValueFunction implements ValueFunctionWithConstant, Serializable
{
	private static final long serialVersionUID = 2801982061205871665L;

	private static PlusMultiply singleObj = null;

	private final double _cnt;
	
	private PlusMultiply() {
		_cnt = 1;
	}
	
	private PlusMultiply(double cnt) {
		_cnt = cnt;
	}

	public static PlusMultiply getFnObject() {
		if ( singleObj == null )
			singleObj = new PlusMultiply();
		return singleObj;
	}
	
	@Override
	public double execute(double in1, double in2, double in3) {
		return in1 + in2 * in3;
	}

	@Override
	public BinaryOperator setOp2Constant(double cnt) {
		return new BinaryOperator(new PlusMultiply(cnt));
	}
	
	@Override
	public double execute(double in1, double in2) {
		return in1 + _cnt * in2;
	}
}
