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


package org.apache.sysds.runtime.matrix.operators;

import java.io.Serializable;

import org.apache.sysds.runtime.functionobjects.IfElse;
import org.apache.sysds.runtime.functionobjects.MinusMultiply;
import org.apache.sysds.runtime.functionobjects.PlusMultiply;
import org.apache.sysds.runtime.functionobjects.TernaryValueFunction;

public class TernaryOperator  extends Operator implements Serializable
{
	private static final long serialVersionUID = 3456088891054083634L;
	
	public final TernaryValueFunction fn;
	
	public TernaryOperator(TernaryValueFunction p) {
		//ternaryop is sparse-safe iff (op 0 0 0) == 0
		super (p instanceof PlusMultiply || p instanceof MinusMultiply || p instanceof IfElse);
		fn = p;
	}
	
	@Override
	public String toString() {
		return "TernaryOperator("+fn.getClass().getSimpleName()+")";
	}
}
