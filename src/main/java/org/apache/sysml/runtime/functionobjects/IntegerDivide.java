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

import org.apache.sysml.runtime.util.UtilFunctions;

public class IntegerDivide extends ValueFunction 
{

	private static final long serialVersionUID = -6994403907602762873L;

	private static IntegerDivide singleObj = null;
	
	private IntegerDivide() {
		// nothing to do here
	}
	
	public static IntegerDivide getIntegerDivideFnObject() {
		if ( singleObj == null )
			singleObj = new IntegerDivide();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	@Override
	public double execute(double in1, double in2) {
		return executeIntDiv( in1, in2 );
	}

	@Override
	public double execute(double in1, long in2) {
		return executeIntDiv( in1, (double)in2 );
	}

	@Override
	public double execute(long in1, double in2) {
		return executeIntDiv( (double)in1, in2 );
	}

	@Override
	public double execute(long in1, long in2) {
		return executeIntDiv( (double)in1, (double)in2 );
	}

	/**
	 * NOTE: The R semantics of integer divide a%/%b are to compute the 
	 * double division and subsequently cast to int. In case of a NaN 
	 * or +-INFINITY devision result, the overall output is NOT cast to
	 * int in order to prevent the special double values.
	 * 
	 * @param in1
	 * @param in2
	 * @return
	 */
	private double executeIntDiv( double in1, double in2 )
	{
		//compute normal double devision
		double ret = in1 / in2;
		
		//check for NaN/+-INF intermediate (cast to int would eliminate it)
		if( Double.isNaN(ret) || Double.isInfinite(ret) )
			return ret;
		
		//safe cast to int
		return UtilFunctions.toLong( ret );
	}
}
