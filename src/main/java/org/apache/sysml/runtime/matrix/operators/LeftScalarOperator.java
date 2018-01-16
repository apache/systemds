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

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.GreaterThan;
import org.apache.sysml.runtime.functionobjects.GreaterThanEquals;
import org.apache.sysml.runtime.functionobjects.LessThan;
import org.apache.sysml.runtime.functionobjects.LessThanEquals;
import org.apache.sysml.runtime.functionobjects.ValueFunction;
import org.apache.sysml.runtime.functionobjects.Builtin.BuiltinCode;

/**
 * Scalar operator for scalar-matrix operations with scalar 
 * on the left-hand-side.
 * 
 */
public class LeftScalarOperator extends ScalarOperator 
{	
	private static final long serialVersionUID = 2360577666575746424L;
	
	public LeftScalarOperator(ValueFunction p, double cst) {
		super(p, cst, (p instanceof GreaterThan && cst<=0)
			|| (p instanceof GreaterThanEquals && cst<0)
			|| (p instanceof LessThan && cst>=0)
			|| (p instanceof LessThanEquals && cst>0)
			|| (Builtin.isBuiltinCode(p, BuiltinCode.MAX) && cst<=0)
			|| (Builtin.isBuiltinCode(p, BuiltinCode.MIN) && cst>=0));
	}
	
	@Override
	public ScalarOperator setConstant(double cst) {
		return new LeftScalarOperator(fn, cst);
	}

	@Override
	public double executeScalar(double in) throws DMLRuntimeException {
		return fn.execute(_constant, in);
	}
}
