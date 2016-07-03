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

public abstract class ScalarObjectFactory
{
	/**
	 * 
	 * @param vt
	 * @param obj
	 * @return
	 */
	public static ScalarObject createScalarObject(ValueType vt, Object obj) {
		switch( vt ) {
			case BOOLEAN: return new BooleanObject((Boolean)obj);
			case INT:     return new IntObject((Long)obj);
			case DOUBLE:  return new DoubleObject((Double)obj);
			case STRING:  return new StringObject((String)obj);
			default: throw new RuntimeException("Unsupported scalar object type: "+vt.toString());
		}
	}
}
