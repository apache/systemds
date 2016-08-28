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

package org.apache.sysml.runtime.functionobjects;

/**
 * Integer modulus, where we adhere to the defined R semantics:
 * 
 * ("%% indicates x mod y and %/% indicates integer division. 
 * It is guaranteed that x == (x %% y) + y * ( x %/% y ) (up to rounding error) 
 * unless y == 0")
 * 
 */
public class Modulus extends ValueFunction 
{

	private static final long serialVersionUID = -1409182981172200840L;

	private static Modulus singleObj = null;
	private IntegerDivide _intdiv = null;
	
	private Modulus() {
		_intdiv = IntegerDivide.getFnObject();
	}
	
	public static Modulus getFnObject() {
		if ( singleObj == null )
			singleObj = new Modulus();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	@Override
	public double execute(double in1, double in2) {
		if( in2==0.0 || in2==-0.0 )
			return Double.NaN;
		return in1 - _intdiv.execute(in1, in2)*in2;
	}

	@Override
	public double execute(double in1, long in2) {
		if( in2==0 )
			return Double.NaN;
		return in1 - _intdiv.execute(in1, in2)*in2;
	}

	@Override
	public double execute(long in1, double in2) {
		if( in2==0.0 || in2==-0.0 )
			return Double.NaN;
		return in1 - _intdiv.execute(in1, in2)*in2;
	}

	@Override
	public double execute(long in1, long in2) {
		if( in2==0 )
			return Double.NaN;
		return in1 - _intdiv.execute(in1, in2)*in2;
	}	
	
}
