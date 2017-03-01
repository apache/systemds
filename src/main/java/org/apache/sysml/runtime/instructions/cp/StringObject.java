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
import org.apache.sysml.runtime.DMLRuntimeException;

public class StringObject extends ScalarObject 
{	
	private static final long serialVersionUID = 2464839775369002455L;

	private static final int MAX_STRING_SIZE = 1*1024*1024; //1MB
	
	private String _value;

	public StringObject (String val){
		this(null,val);
	}

	public StringObject(String name, String val){
		super(name, ValueType.STRING);
		_value = val;
	}
	
	@Override
	public boolean getBooleanValue() {
		return "TRUE".equals(_value);
	}

	@Override
	public long getLongValue(){
		return getBooleanValue() ? 1 : 0;
	}

	@Override
	public double getDoubleValue(){
		return getBooleanValue() ? 1d : 0d;
	}

	@Override
	public String getStringValue(){
		return _value;
	}

	@Override
	public Object getValue(){
		return _value;
	}

	public static void checkMaxStringLength( long len ) 
		throws DMLRuntimeException
	{
		if( len > MAX_STRING_SIZE ) {
			throw new DMLRuntimeException("Output string length exceeds maximum "
				+ "scalar string length ("+len+" > "+MAX_STRING_SIZE+").");
		}
	}
}
