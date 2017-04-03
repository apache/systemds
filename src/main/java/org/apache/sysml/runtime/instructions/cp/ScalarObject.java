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

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;

public abstract class ScalarObject extends Data
{
	private static final long serialVersionUID = 6994413375932824892L;

	private String _name;
	
	public ScalarObject(String name, ValueType vt) {
		super(DataType.SCALAR, vt);
		_name = name;
	}

	public String getName() {
		return _name;
	}

	public abstract boolean getBooleanValue();

	public abstract long getLongValue();
	
	public abstract double getDoubleValue();

	public abstract String getStringValue();
	
	public String getLanguageSpecificStringValue() {
		return getStringValue();
	}
	
	public abstract Object getValue();	
	
	@Override
	public String toString() { 
		return getStringValue();
	}
	
	@Override
	public String getDebugName() {
		return getStringValue();
	}
}
