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

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.parser.Expression.ValueType;


public class BooleanObject extends ScalarObject  
{

	private static final long serialVersionUID = -4506242165735516984L;

	private boolean _value;

	public BooleanObject(boolean val){
		this(null,val);
	}

	public BooleanObject(String name,boolean val){
		super(name, ValueType.BOOLEAN);
		_value = val;
	}

	@Override
	public boolean getBooleanValue(){
		return _value;
	}
	
	@Override
	public long getLongValue() {
		return _value ? 1 : 0;
	}

	@Override
	public double getDoubleValue(){
		return _value ? 1d : 0d;
	}

	@Override
	public String getStringValue(){
		return Boolean.toString(_value).toUpperCase();
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
