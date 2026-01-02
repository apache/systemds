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

package org.apache.sysds.runtime.instructions.cp;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.CMOperator.AggregateOperationTypes;


public class CmCovObject extends Data 
{
	private static final long serialVersionUID = -5814207545197934085L;

	//for central moment
	public double w;
	public KahanObject mean;
	public KahanObject m2;
	public KahanObject m3;
	public KahanObject m4;
	public double min;
	public double max;
	
	public KahanObject mean_v;
	public KahanObject c2;
	
	@Override
	public String toString() {
		return "weight: "+w+", mean: "+mean+", m2: "+m2+", m3: "+m3+", m4: "+m4+", min: "+min+", max: "+max+", mean2: "+mean_v+", c2: "+c2;
	}
	
	public CmCovObject()
	{
		super(DataType.UNKNOWN, ValueType.UNKNOWN);
		w=0;
		mean=new KahanObject(0,0);
		m2=new KahanObject(0,0);
		m3=new KahanObject(0,0);
		m4=new KahanObject(0,0);
		mean_v=new KahanObject(0,0);
		c2=new KahanObject(0,0);
		min=0;
		max=0;
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
		min=0;
		max=0;
	}
	
	public int compareTo(CmCovObject that)
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
		else if(min!=that.min)
			return Double.compare(min, that.min);
		else if(max!=that.max)
			return Double.compare(max, that.max);
		else
			return KahanObject.compare(c2, that.c2);
	}
	
	@Override
	public boolean equals(Object o)
	{
		if( o == null || !(o instanceof CmCovObject) )
			return false;
		
		CmCovObject that = (CmCovObject)o;
		return (w==that.w && mean.equals(that.mean) && m2.equals(that.m2))
				&& m3.equals(that.m3) && m4.equals(that.m4) 
				&& mean_v.equals(that.mean_v) && c2.equals(that.c2)
				&& min==that.min && max == that.max;
	}
	
	@Override
	public int hashCode() {
		throw new RuntimeException("hashCode() should never be called on instances of this class.");
	}
	
	public void set(CmCovObject that)
	{
		this.w=that.w;
		this.mean.set(that.mean);
		this.m2.set(that.m2);
		this.m3.set(that.m3);
		this.m4.set(that.m4);
		this.mean_v.set(that.mean_v);
		this.c2.set(that.c2);
		this.min=that.min;
		this.max=that.max;
	}
	
	public boolean isCMAllZeros()
	{
		return w==0 && mean.isAllZero() && m2.isAllZero()  && m3.isAllZero()  && m4.isAllZero() && min==0 && max==0;
	}
	
	public boolean isCOVAllZeros()
	{
		return w==0 && mean.isAllZero()  && mean_v.isAllZero() && c2.isAllZero() ;
	}

	/**
	 * Return the result of the aggregated operation given the
	 * operator.
	 * 
	 * @param op operator
	 * @return result of the aggregated operation for the given operator
	 */
	public double getRequiredResult(Operator op) {
		if(op instanceof CMOperator) {
			AggregateOperationTypes agg=((CMOperator)op).aggOpType;
			return getRequiredResult(agg);
		}
		else {
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
	 * 
	 * @param agg aggregate operation type
	 * @return result of the aggregated operation given the operation type
	 */
	public double getRequiredResult(AggregateOperationTypes agg) {
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
			case MIN:
				return min;
			case MAX:
				return max;
			case VARIANCE:
				return w==1.0? 0:m2._sum/(w-1);
			default:
				throw new DMLRuntimeException("Invalid aggreagte in CM_CV_Object: " + agg);
		}
	}

	public double getRequiredPartialResult(Operator op) {
		if(op instanceof CMOperator) {
			AggregateOperationTypes agg=((CMOperator)op).aggOpType;
			switch(agg) {
				case COUNT: return 0;
				case MEAN:  return mean._sum;
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

	public double getWeight() {
		return w;
	}
	
	@Override
	public String getDebugName() {
		return "CM_COV_"+hashCode();
	}
}
