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

public class Not extends ValueFunction 
{
	private static final long serialVersionUID = -3137270229584267263L;

	private static Not singleObj = null;

	private Not() {
		// nothing to do here
	}
	
	public static Not getNotFnObject() {
		if ( singleObj == null )
			singleObj = new Not();
		return singleObj;
	}

	@Override
	public boolean execute(boolean in) {
		return !in;
	}

	@Override
	public double execute(double in) {
		return (in == 0) ? 1 : 0;
	}

	@Override
	public boolean isBinary(){
		return true;
	}
}
