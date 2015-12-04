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

package org.apache.sysml.runtime.instructions.cp;


import org.apache.sysml.parser.Expression.ValueType;

public class IntObject extends ScalarObject
{

	private static final long serialVersionUID = 353170585998999528L;

	//we use consistently to the compiler long in terms of integer (8 byte)
	private long _value;

	public IntObject(long val)
	{
		this(null,val);
	}

	public IntObject(String name, long val)
	{
		super(name, ValueType.INT);
		_value = val;
	}

	@Override
	public boolean getBooleanValue(){
		return (_value!=0);
	}
	
	@Override
	public long getLongValue(){
		return _value;
	}
	
	@Override
	public double getDoubleValue(){
		return (double) _value;
	}

	@Override
	public String getStringValue(){
		return Long.toString(_value);
	}
	
	@Override
	public Object getValue(){
		return _value;
	}

	public String toString() { 
		return getStringValue();
	}

	@Override
	public String getDebugName() {
		return null;
	}
}
