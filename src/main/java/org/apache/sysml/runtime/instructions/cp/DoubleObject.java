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
import org.apache.sysml.runtime.util.UtilFunctions;

public class DoubleObject extends ScalarObject 
{
	private static final long serialVersionUID = -8525290101679236360L;

	private double _value;

	public DoubleObject(double val){
		this(null,val);
	}

	public DoubleObject(String name, double val){
		super(name, ValueType.DOUBLE);
		_value = val;
	}

	@Override
	public boolean getBooleanValue(){
		return (_value != 0);
	}

	@Override
	public long getLongValue() {
		return UtilFunctions.toLong(_value);
	}
	
	@Override
	public double getDoubleValue(){
		return _value;
	}
	
	@Override
	public String getStringValue(){
		return Double.toString(_value);
	}
	
	@Override
	public Object getValue(){
		return _value;
	}

	@Override
	public String getDebugName() {
		return null;
	}
}
