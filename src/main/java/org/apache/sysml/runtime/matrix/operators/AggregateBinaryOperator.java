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


package org.apache.sysml.runtime.matrix.operators;

import java.io.Serializable;

import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.functionobjects.Plus;
import org.apache.sysml.runtime.functionobjects.ValueFunction;


public class AggregateBinaryOperator extends Operator implements Serializable
{

	private static final long serialVersionUID = 1666421325090925726L;

	public ValueFunction binaryFn;
	public AggregateOperator aggOp;
	private int k; //num threads
	
	public AggregateBinaryOperator(ValueFunction inner, AggregateOperator outer)
	{
		//default degree of parallelism is 1 
		//(for example in MR/Spark because we parallelize over the number of blocks)
		this( inner, outer, 1 );
	}
	
	public AggregateBinaryOperator(ValueFunction inner, AggregateOperator outer, int numThreads)
	{
		binaryFn = inner;
		aggOp = outer;
		k = numThreads;
		
		//so far, we only support matrix multiplication, and it is sparseSafe
		if(binaryFn instanceof Multiply && aggOp.increOp.fn instanceof Plus)
			sparseSafe=true;
		else
			sparseSafe=false;
	}
	
	public int getNumThreads() {
		return k;
	}
}
