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


package com.ibm.bi.dml.runtime.matrix.operators;

import com.ibm.bi.dml.runtime.functionobjects.FunctionObject;

/*
 * Simple operator is just a wrapper for a single function object of any type.
 */
public class SimpleOperator extends Operator 
{

	private static final long serialVersionUID = 625147299273287379L;

	public FunctionObject fn;
	
	public SimpleOperator ( FunctionObject f ) {
		fn = f;
	}
}
