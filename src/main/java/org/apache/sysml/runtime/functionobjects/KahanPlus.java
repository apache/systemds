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

import java.io.Serializable;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.KahanObject;


public class KahanPlus extends KahanFunction implements Serializable
{

	private static final long serialVersionUID = -8338160609569967791L;

	private static KahanPlus singleObj = null;
	
	private KahanPlus() {
		// nothing to do here
	}
	
	public static KahanPlus getKahanPlusFnObject() {
		if ( singleObj == null )
			singleObj = new KahanPlus();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	@Override
	public Data execute(Data in1, double in2) 
		throws DMLRuntimeException 
	{
		KahanObject kahanObj=(KahanObject)in1;
		
		//fast path for INF/-INF in order to ensure result correctness
		//(computing corrections otherwise incorrectly computes NaN)
		if( Double.isInfinite(kahanObj._sum) || Double.isInfinite(in2) ) {
			kahanObj.set(Double.isInfinite(in2) ? in2 : kahanObj._sum, 0);
			return kahanObj;
		}
		
		//default path for any other value
		double correction=in2+kahanObj._correction;
		double sum=kahanObj._sum+correction;
		kahanObj.set(sum, correction-(sum-kahanObj._sum)); //prevent eager JIT opt 		
		return kahanObj;
	}
	
	@Override // in1, in2 is the sum, in3 is the correction
	public Data execute(Data in1, double in2, double in3) 
		throws DMLRuntimeException 
	{
		KahanObject kahanObj=(KahanObject)in1;
		
		//fast path for INF/-INF in order to ensure result correctness
		//(computing corrections otherwise incorrectly computes NaN)
		if( Double.isInfinite(kahanObj._sum) || Double.isInfinite(in2) ) {
			kahanObj.set(Double.isInfinite(in2) ? in2 : kahanObj._sum, 0);
			return kahanObj;
		}
		
		//default path for any other value
		double correction=in2+(kahanObj._correction+in3);
		double sum=kahanObj._sum+correction;
		kahanObj.set(sum, correction-(sum-kahanObj._sum)); //prevent eager JIT opt
		return kahanObj;
	}
	
	/**
	 * Simplified version of execute(Data in1, double in2) 
	 * without exception handling and casts.
	 * 
	 * @param in1
	 * @param in2
	 */
	public void execute2(KahanObject in1, double in2) 
	{
		//fast path for INF/-INF in order to ensure result correctness
		//(computing corrections otherwise incorrectly computes NaN)
		if( Double.isInfinite(in1._sum) || Double.isInfinite(in2) ) {
			in1.set(Double.isInfinite(in2) ? in2 : in1._sum, 0);
			return;
		}
		
		//default path for any other value
		double correction = in2 + in1._correction;
		double sum = in1._sum + correction;
		in1.set(sum, correction-(sum-in1._sum)); //prevent eager JIT opt 	
	}
	
	@Override
	public void execute3(KahanObject in1, double in2, int count) {
		execute2(in1, in2*count);
	}
}
