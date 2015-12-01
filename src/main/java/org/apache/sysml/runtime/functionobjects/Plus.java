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

import java.io.Serializable;

import org.apache.sysml.runtime.DMLRuntimeException;

// Singleton class

public class Plus extends ValueFunction implements Serializable 
{

	private static final long serialVersionUID = -3573790367761963555L;

	private static Plus singleObj = null;
	
	private Plus() {
		// nothing to do here
	}
	
	public static Plus getPlusFnObject() {
		if ( singleObj == null )
			singleObj = new Plus();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	@Override
	public double execute(double in1, double in2) {
		return in1 + in2;
	}

	@Override
	public double execute(double in1, long in2) {
		return in1 + in2;
	}

	@Override
	public double execute(long in1, double in2) {
		return in1 + in2;
	}

	@Override
	public double execute(long in1, long in2) {
		//for robustness regarding long overflows (only used for scalar instructions)
		double dval = ((double)in1 + in2);
		if( dval > Long.MAX_VALUE )
			return dval;	
		
		return in1 + in2;
	}

	public String execute ( String in1, String in2 ) throws DMLRuntimeException {
		return in1 + in2;
	}
	
}
