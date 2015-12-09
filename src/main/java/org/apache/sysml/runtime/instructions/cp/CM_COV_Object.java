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

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.operators.CMOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.CMOperator.AggregateOperationTypes;


public class CM_COV_Object extends Data 
{

	private static final long serialVersionUID = -5814207545197934085L;

	//for central moment
	public double w;
	public KahanObject mean;
	public KahanObject m2;
	public KahanObject m3;
	public KahanObject m4;
	
	public KahanObject mean_v;
	public KahanObject c2;
	
	public String toString()
	{
		return "weight: "+w+", mean: "+mean+", m2: "+m2+", m3: "+m3+", m4: "+m4+", mean2: "+mean_v+", c2: "+c2;
	}
	
	public CM_COV_Object()
	{
		super(DataType.OBJECT, ValueType.UNKNOWN);
		w=0;
		mean=new KahanObject(0,0);
		m2=new KahanObject(0,0);
		m3=new KahanObject(0,0);
		m4=new KahanObject(0,0);
		mean_v=new KahanObject(0,0);
		c2=new KahanObject(0,0);
	}
	
	public void reset()
	{
		w=0;
		mean=new KahanObject(0,0);
		m2=new KahanObject(0,0);
		m3=new KahanObject(0,0);
		m4=new KahanObject(0,0);
		mean_v=new KahanObject(0,0);
		c2=new KahanObject(0,0);
	}
	
	public int compareTo(CM_COV_Object that)
	{
		if(w!=that.w)
			return Double.compare(w, that.w);
		else if(mean!=that.mean)
			return KahanObject.compare(mean, that.mean);
		else if(m2!=that.m2)
			return KahanObject.compare(m2, that.m2);
		else if(m3!=that.m3)
			return KahanObject.compare(m3, that.m3);
		else if(m4!=that.m4)
			return KahanObject.compare(m4, that.m4);
		else if(mean_v!=that.mean_v)
			return KahanObject.compare(mean_v, that.mean_v);
		else
			return KahanObject.compare(c2, that.c2);
	}
	
	@Override
	public boolean equals(Object o)
	{
		if( o == null || !(o instanceof CM_COV_Object) )
			return false;
		
		CM_COV_Object that = (CM_COV_Object)o;
		return (w==that.w && mean.equals(that.mean) && m2.equals(that.m2))
				&& m3.equals(that.m3) && m4.equals(that.m4) 
				&& mean_v.equals(that.mean_v) && c2.equals(that.c2);
	}
	
	@Override
	public int hashCode() {
		throw new RuntimeException("hashCode() should never be called on instances of this class.");
	}
	
	public void set(CM_COV_Object that)
	{
		this.w=that.w;
		this.mean.set(that.mean);
		this.m2.set(that.m2);
		this.m3.set(that.m3);
		this.m4.set(that.m4);
		this.mean_v.set(that.mean_v);
		this.c2.set(that.c2);
	}
	
	public boolean isCMAllZeros()
	{
		return w==0 && mean.isAllZero() && m2.isAllZero()  && m3.isAllZero()  && m4.isAllZero() ;
	}
	
	public boolean isCOVAllZeros()
	{
		return w==0 && mean.isAllZero()  && mean_v.isAllZero() && c2.isAllZero() ;
	}

	/**
	 * Return the result of the aggregated operation given the
	 * operator.
	 */
	public double getRequiredResult(Operator op) throws DMLRuntimeException
	{
		if(op instanceof CMOperator)
		{
			AggregateOperationTypes agg=((CMOperator)op).aggOpType;
			return getRequiredResult(agg);
		}
		else
		{
			//avoid division by 0
			if(w==1.0)
				return 0;
			else
				return c2._sum/(w-1.0);
		}
	}

	/**
	 * Return the result of the aggregated operation given the
	 * operation type.
	 */
	public double getRequiredResult(AggregateOperationTypes agg) throws DMLRuntimeException {
		switch(agg)
		{
			case COUNT:
				return w;
			case MEAN:
				return mean._sum;
			case CM2:
				return m2._sum/w;
			case CM3:
				return m3._sum/w;
			case CM4:
				return m4._sum/w;
			case VARIANCE:
				return w==1.0? 0:m2._sum/(w-1);
			default:
				throw new DMLRuntimeException("Invalid aggreagte in CM_CV_Object: " + agg);
		}
	}

	/**
	 * 
	 * @param op
	 * @return
	 * @throws DMLRuntimeException
	 */
	public double getRequiredPartialResult(Operator op) 
		throws DMLRuntimeException
	{
		if(op instanceof CMOperator)
		{
			AggregateOperationTypes agg=((CMOperator)op).aggOpType;
			switch(agg)
			{
				case COUNT:
					return 0;
				case MEAN:
					return mean._sum;
				case CM2:					
				case CM3:					
				case CM4:
				case VARIANCE:
					throw new DMLRuntimeException("Aggregation operator '"+agg.toString()+"' does not apply to partial aggregation.");
				default:
					throw new DMLRuntimeException("Invalid aggreagte in CM_CV_Object: " + agg);
			}
		}
		else
			return c2._sum;
	}

	/**
	 * 
	 * @return
	 */
	public double getWeight() 
	{
		return w;
	}
	
	@Override
	public String getDebugName() {
		return "CM_COV_"+hashCode();
	}
}
