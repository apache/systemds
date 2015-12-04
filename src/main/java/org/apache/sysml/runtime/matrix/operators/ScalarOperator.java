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
import org.apache.sysml.runtime.functionobjects.And;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.Builtin.BuiltinFunctionCode;
import org.apache.sysml.runtime.functionobjects.Equals;
import org.apache.sysml.runtime.functionobjects.GreaterThan;
import org.apache.sysml.runtime.functionobjects.LessThan;
import org.apache.sysml.runtime.functionobjects.Minus;
import org.apache.sysml.runtime.functionobjects.MinusNz;
import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.functionobjects.Multiply2;
import org.apache.sysml.runtime.functionobjects.NotEquals;
import org.apache.sysml.runtime.functionobjects.Power;
import org.apache.sysml.runtime.functionobjects.Power2;
import org.apache.sysml.runtime.functionobjects.ValueFunction;


public class ScalarOperator  extends Operator 
{
	private static final long serialVersionUID = 4547253761093455869L;

	
	public ValueFunction fn;
	protected double _constant;
	
	public ScalarOperator(ValueFunction p, double cst)
	{
		fn = p;
		_constant = cst;
		
		//as long as (0 op v)=0, then op is sparsesafe
		//note: additional functionobjects might qualify according to constant
		if(   fn instanceof Multiply || fn instanceof Multiply2 
		   || fn instanceof Power || fn instanceof Power2 
		   || fn instanceof And || fn instanceof MinusNz
		   || (fn instanceof Builtin && ((Builtin)fn).getBuiltinFunctionCode()==BuiltinFunctionCode.LOG_NZ)) 
		{
			sparseSafe=true;
		}
		else
		{
			sparseSafe=false;
		}
	}
	
	public double getConstant()
	{
		return _constant;
	}
	
	public void setConstant(double cst) 
	{
		//set constant
		_constant = cst;
		
		//revisit sparse safe decision according to known constant
		//note: there would be even more potential if we take left/right op into account
		if(    fn instanceof Multiply || fn instanceof Multiply2 
			|| fn instanceof Power || fn instanceof Power2 
			|| fn instanceof And || fn instanceof MinusNz
			|| fn instanceof Builtin && ((Builtin)fn).getBuiltinFunctionCode()==BuiltinFunctionCode.LOG_NZ
			|| (fn instanceof GreaterThan && _constant==0) 
			|| (fn instanceof LessThan && _constant==0)
			|| (fn instanceof NotEquals && _constant==0)
			|| (fn instanceof Equals && _constant!=0)
			|| (fn instanceof Minus && _constant==0))
		{
			sparseSafe = true;
		}
		else
		{
			sparseSafe = false;
		}
	}
	
	public double executeScalar(double in) throws DMLRuntimeException {
		throw new DMLRuntimeException("executeScalar(): can not be invoked from base class.");
	}
}
