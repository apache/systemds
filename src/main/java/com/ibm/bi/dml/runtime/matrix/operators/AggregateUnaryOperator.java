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

import com.ibm.bi.dml.runtime.functionobjects.IndexFunction;
import com.ibm.bi.dml.runtime.functionobjects.KahanPlus;
import com.ibm.bi.dml.runtime.functionobjects.Minus;
import com.ibm.bi.dml.runtime.functionobjects.Or;
import com.ibm.bi.dml.runtime.functionobjects.Plus;


public class AggregateUnaryOperator  extends Operator 
{

	private static final long serialVersionUID = 6690553323120787735L;

	public AggregateOperator aggOp;
	public IndexFunction indexFn;
	
	public AggregateUnaryOperator(AggregateOperator aop, IndexFunction iop)
	{
		aggOp=aop;
		indexFn=iop;
		
		//decide on sparse safe
		if( aggOp.increOp.fn instanceof Plus || 
			aggOp.increOp.fn instanceof KahanPlus || 
			aggOp.increOp.fn instanceof Or || 
			aggOp.increOp.fn instanceof Minus ) 
		{
			sparseSafe=true;
		}
		else
			sparseSafe=false;
	}
}
