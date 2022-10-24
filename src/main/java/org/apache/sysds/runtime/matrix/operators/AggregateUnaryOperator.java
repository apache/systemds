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

import org.apache.sysds.runtime.functionobjects.IndexFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Or;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceRow;


public class AggregateUnaryOperator extends MultiThreadedOperator {
	private static final long serialVersionUID = 6690553323120787735L;

	public final AggregateOperator aggOp;
	public final IndexFunction indexFn;

	public AggregateUnaryOperator(AggregateOperator aop, IndexFunction iop)
	{
		//default degree of parallelism is 1 
		//(for example in MR/Spark because we parallelize over the number of blocks)
		this( aop, iop, 1 );
	}

	public AggregateUnaryOperator(AggregateOperator aop, IndexFunction iop, int numThreads)
	{
		super(aop.increOp.fn instanceof Plus 
			|| aop.increOp.fn instanceof KahanPlus 
			|| aop.increOp.fn instanceof KahanPlusSq 
			|| aop.increOp.fn instanceof Or 
			|| aop.increOp.fn instanceof Minus);
		aggOp = aop;
		indexFn = iop;
		_numThreads = numThreads;
	}

	public boolean isRowAggregate() {
		return indexFn instanceof ReduceCol;
	}
	
	public boolean isColAggregate() {
		return indexFn instanceof ReduceRow;
	}

	public boolean isFullAggregate() {
		return indexFn instanceof ReduceAll;
	}

	@Override
	public String toString(){
		StringBuilder sb = new StringBuilder();
		sb.append(getClass().getSimpleName());
		sb.append("(");
		sb.append(aggOp);
		sb.append(", ");
		sb.append(indexFn);
		sb.append(")");
		return sb.toString();
	}
}
