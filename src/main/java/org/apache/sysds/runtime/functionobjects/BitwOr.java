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

import org.apache.sysds.runtime.util.UtilFunctions;

public class BitwOr extends ValueFunction implements Serializable
{
	private static final long serialVersionUID = -9172774392245257468L;

	private static BitwOr singleObj = null;

	private BitwOr() {
		// nothing to do here
	}

	public static BitwOr getBitwOrFnObject() {
		if ( singleObj == null )
			singleObj = new BitwOr();
		return singleObj;
	}
	
	@Override
	public double execute(long in1, long in2) {
		return (int)in1 | (int)in2;
	}

	@Override
	public double execute(double in1, double in2) {
		return UtilFunctions.toInt(in1) | UtilFunctions.toInt(in2);
	}
}
