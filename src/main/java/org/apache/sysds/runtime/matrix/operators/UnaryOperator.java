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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.ValueFunction;

public class UnaryOperator extends MultiThreadedOperator
{
	private static final long serialVersionUID = 2441990876648978637L;

	public final ValueFunction fn;
	private final boolean inplace;

	public UnaryOperator(ValueFunction p) {
		this(p, 1, false); //default single-threaded
	}
	
	public UnaryOperator(ValueFunction p, int numThreads, boolean inPlace) {
		super(p instanceof Builtin &&
			(((Builtin)p).bFunc==Builtin.BuiltinCode.SIN || ((Builtin)p).bFunc==Builtin.BuiltinCode.TAN 
			// sinh and tanh are zero only at zero, else they are nnz
			|| ((Builtin)p).bFunc==Builtin.BuiltinCode.SINH || ((Builtin)p).bFunc==Builtin.BuiltinCode.TANH
			|| ((Builtin)p).bFunc==Builtin.BuiltinCode.ROUND || ((Builtin)p).bFunc==Builtin.BuiltinCode.ABS
			|| ((Builtin)p).bFunc==Builtin.BuiltinCode.SQRT || ((Builtin)p).bFunc==Builtin.BuiltinCode.SPROP
			|| ((Builtin)p).bFunc==Builtin.BuiltinCode.LOG_NZ || ((Builtin)p).bFunc==Builtin.BuiltinCode.SIGN) );
		fn = p;
		_numThreads = numThreads;
		inplace = inPlace;
	}
	
	public boolean isInplace() {
		return inplace;
	}
	
	public double getPattern() {
		switch( ((Builtin)fn).bFunc ) {
			case ISNAN:
			case ISNA:   return Double.NaN;
			case ISINF:  return Double.POSITIVE_INFINITY;
			default:
				throw new DMLRuntimeException(
					"No pattern existing for "+((Builtin)fn).bFunc.name());
		}
	}
}
