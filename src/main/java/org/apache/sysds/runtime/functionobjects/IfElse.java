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

package org.apache.sysds.runtime.functionobjects;

import java.io.Serializable;

public class IfElse extends TernaryValueFunction implements Serializable
{
	private static final long serialVersionUID = -8660124936856173978L;
	
	private static IfElse singleObj = null;

	private IfElse() {
		// nothing to do here
	}

	public static IfElse getFnObject() {
		if ( singleObj == null )
			singleObj = new IfElse();
		return singleObj;
	}
	
	@Override
	public double execute(double in1, double in2, double in3) {
		return (in1 != 0) ? in2 : in3;
	}
}
