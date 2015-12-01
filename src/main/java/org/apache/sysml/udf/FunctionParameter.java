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

package com.ibm.bi.dml.udf;

import java.io.Serializable;

/**
 * abstract class to represent all input and output objects for package
 * functions.
 * 
 * 
 * 
 */

public abstract class FunctionParameter implements Serializable
{
	

	private static final long serialVersionUID = 1189133371204708466L;
	
	public enum FunctionParameterType{
		Matrix, 
		Scalar, 
		Object,
	}
	
	private FunctionParameterType _type;

	/**
	 * Constructor to set type
	 * 
	 * @param type
	 */
	public FunctionParameter(FunctionParameterType type) {
		_type = type;
	}

	/**
	 * Method to get type
	 * 
	 * @return
	 */
	public FunctionParameterType getType() {
		return _type;
	}

}
