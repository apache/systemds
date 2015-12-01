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
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;

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
	
	public String toString() { 
		return getStringValue();
	}

	@Override
	public String getDebugName() {
		return _value;
	}

	/**
	 * 
	 * @param len
	 * @throws DMLUnsupportedOperationException
	 */
	public static void checkMaxStringLength( long len ) 
		throws DMLRuntimeException
	{
		if( len > MAX_STRING_SIZE )
		{
			throw new DMLRuntimeException(
					"Output string length exceeds maximum scalar string length " +
					"("+len+" > "+MAX_STRING_SIZE+").");
		}
	}
}
