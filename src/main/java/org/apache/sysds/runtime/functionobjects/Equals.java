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


public class Equals extends ValueComparisonFunction
{
	private static final long serialVersionUID = -8887713112454357802L;

	private static Equals singleObj = null;

	private Equals() {
		// nothing to do here
	}
	
	public static Equals getEqualsFnObject() {
		if ( singleObj == null )
			singleObj = new Equals();
		return singleObj;
	}

	/*
	 * Arithmetic relational operators (==, !=, <=, >=) must be instead of
	 * <code>Double.compare()</code> due to the inconsistencies in the way
	 * NaN and -0.0 are handled. The behavior of methods in
	 * <code>Double</code> class are designed mainly to make Java
	 * collections work properly. For more details, see the help for
	 * <code>Double.equals()</code> and <code>Double.compareTo()</code>.
	 */
	
	/**
	 * execute() method that returns double is required since current map-reduce
	 * runtime can only produce matrices of doubles. This method is used on MR
	 * side to perform comparisons on matrices like A==B and A==2.5
	 */
	@Override
	public double execute(double in1, double in2) {
		return (in1 == in2 ? 1.0 : 0.0);
	}
	
	@Override
	public boolean compare(double in1, double in2) {
		return (in1 == in2);
	}

	@Override
	public boolean compare(long in1, long in2) {
		return (in1 == in2);
	}
	
	@Override
	public boolean compare(boolean in1, boolean in2) {
		return (in1 == in2);
	}

	@Override
	public boolean compare(String in1, String in2) {
		return ( in1!=null && in1.equals(in2) );
	}

	@Override
	public boolean isBinary(){
		return true;
	}
}
