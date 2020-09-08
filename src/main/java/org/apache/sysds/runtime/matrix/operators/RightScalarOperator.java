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

import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.GreaterThan;
import org.apache.sysds.runtime.functionobjects.GreaterThanEquals;
import org.apache.sysds.runtime.functionobjects.LessThan;
import org.apache.sysds.runtime.functionobjects.LessThanEquals;
import org.apache.sysds.runtime.functionobjects.Power;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;

/**
 * Scalar operator for scalar-matrix operations with scalar 
 * on the right-hand-side.
 */
public class RightScalarOperator extends ScalarOperator 
{
	private static final long serialVersionUID = 5148300801904349919L;
	
	public RightScalarOperator(ValueFunction p, double cst) {
		this(p, cst, 1);
	}

	public RightScalarOperator(ValueFunction p, double cst, int numThreads){
		super(p, cst, (p instanceof GreaterThan && cst>=0)
			|| (p instanceof GreaterThanEquals && cst>0)
			|| (p instanceof LessThan && cst<=0)
			|| (p instanceof LessThanEquals && cst<0)
			|| (p instanceof Divide && cst!=0)
			|| (p instanceof Power && cst!=0)
			|| (Builtin.isBuiltinCode(p, BuiltinCode.MAX) && cst<=0)
			|| (Builtin.isBuiltinCode(p, BuiltinCode.MIN) && cst>=0), 
			numThreads);
	}

	@Override
	public ScalarOperator setConstant(double cst) {
		return new RightScalarOperator(fn, cst);
	}

	@Override
	public ScalarOperator setConstant(double cst, int numThreads) {
		return new RightScalarOperator(fn, cst, numThreads);
	}
	
	@Override
	public double executeScalar(double in) {
		return fn.execute(in, _constant);
	}
}
