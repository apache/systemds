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

package org.apache.sysml.runtime.functionobjects;

public class GreaterThanEquals extends ValueFunction 
{

	private static final long serialVersionUID = -5444900552418046584L;

	private static GreaterThanEquals singleObj = null;

	private GreaterThanEquals() {
		// nothing to do here
	}
	
	public static GreaterThanEquals getGreaterThanEqualsFnObject() {
		if ( singleObj == null )
			singleObj = new GreaterThanEquals();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	/*
	 * Arithmetic relational operators (==, !=, <=, >=) must be instead of
	 * <code>Double.compare()</code> due to the inconsistencies in the way
	 * NaN and -0.0 are handled. The behavior of methods in
	 * <code>Double</code> class are designed mainly to make Java
	 * collections work properly. For more details, see the help for
	 * <code>Double.equals()</code> and <code>Double.comapreTo()</code>.
	 */
	
	/**
	 * execute() method that returns double is required since current map-reduce
	 * runtime can only produce matrices of doubles. This method is used on MR
	 * side to perform comparisons on matrices like A>=B and A>=2.5
	 */
	@Override
	public double execute(double in1, double in2) {
		return (in1 >= in2 ? 1.0 : 0.0);
	}
	
	@Override
	public boolean compare(double in1, double in2) {
		return (in1 >= in2);
	}

	@Override
	public boolean compare(long in1, long in2) {
		return (in1 >= in2);
	}

	@Override
	public boolean compare(double in1, long in2) {
		return (in1 >= in2);
	}

	@Override
	public boolean compare(long in1, double in2) {
		return (in1 >= in2);
	}
	
	@Override
	public boolean compare(String in1, String in2) {
		return (in1!=null && in1.compareTo(in2)>=0 );
	}
}
