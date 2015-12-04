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

public class Multiply2 extends ValueFunction 
{

	private static final long serialVersionUID = -3762789087715600938L;

	private static Multiply2 singleObj = null;
	
	private Multiply2() {
		// nothing to do here
	}
	
	public static Multiply2 getMultiply2FnObject() {
		if ( singleObj == null )
			singleObj = new Multiply2();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	@Override
	public double execute(double in1) {
		return in1 + in1; //ignore in2 because always 2; 
	}
	
	@Override
	public double execute(double in1, double in2) {
		return in1 + in1; //ignore in2 because always 2; 
	}

	@Override
	public double execute(double in1, long in2) {
		return in1 + in1; //ignore in2 because always 2; 
	}

	@Override
	public double execute(long in1, double in2) {
		return in1 + in1; //ignore in2 because always 2; 
	}

	@Override
	public double execute(long in1, long in2) {
		//for robustness regarding long overflows (only used for scalar instructions)
		double dval = ((double)in1 + in2);
		if( dval > Long.MAX_VALUE )
			return dval;
		
		return in1 + in1; //ignore in2 because always 2; 
	}
}
