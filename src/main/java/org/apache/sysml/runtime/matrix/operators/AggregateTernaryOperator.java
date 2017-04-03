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


package org.apache.sysml.runtime.matrix.operators;

import java.io.Serializable;

import org.apache.sysml.runtime.functionobjects.IndexFunction;
import org.apache.sysml.runtime.functionobjects.ValueFunction;


public class AggregateTernaryOperator extends Operator implements Serializable
{
	private static final long serialVersionUID = 4251745081160216784L;
	
	public ValueFunction binaryFn;
	public AggregateOperator aggOp;
	public IndexFunction indexFn;
	private int k; //num threads
	
	public AggregateTernaryOperator(ValueFunction inner, AggregateOperator outer, IndexFunction ixfun) {
		//default degree of parallelism is 1 (e.g., for distributed operations)
		this( inner, outer, ixfun, 1 );
	}
	
	public AggregateTernaryOperator(ValueFunction inner, AggregateOperator outer, IndexFunction ixfun, int numThreads)
	{
		binaryFn = inner;
		aggOp = outer;
		indexFn = ixfun;
		k = numThreads;
		
		//so far we only support sum-product and its sparse-safe
		sparseSafe = true;
	}
	
	public int getNumThreads() {
		return k;
	}
}
