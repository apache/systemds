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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.operators.CMOperator.AggregateOperationTypes;


/**
 * GENERAL NOTE:
 * * 05/28/2014: We decided to do handle weights consistently to SPSS in an operation-specific manner, 
 *   i.e., we (1) round instead of casting where required (e.g. count), and (2) consistently use
 *   fractional weight values elsewhere. In case a count-base interpretation of weights is needed, just 
 *   ensure rounding before calling CM/COV/KahanPlus.
 * 
 */
public class CM extends ValueFunction 
{

	private static final long serialVersionUID = 9177194651533064123L;

	private AggregateOperationTypes _type = null;
	
	//helper function objects for specific types
	private KahanPlus _plus = null;
	private KahanObject _buff2 = null;
	private KahanObject _buff3 = null;
	
	
	private CM( AggregateOperationTypes type ) 
	{
		_type = type;
		
		switch( _type ) //helper obj on demand
		{
			case COUNT:
				break;
			case CM4:
			case CM3:
				_buff3 = new KahanObject(0, 0);
			case CM2:
				_buff2 = new KahanObject(0, 0);
			case VARIANCE:
			case MEAN:
				_plus = KahanPlus.getKahanPlusFnObject();
				break;
			default:
				//do nothing
		}
	}
	
	public static CM getCMFnObject( AggregateOperationTypes type ) {
		//return new obj, required for correctness in multi-threaded
		//execution due to state in cm object (buff2, buff3)	
		return new CM( type ); 
	}
	
	public static CM getCMFnObject(CM fn) {
		return getCMFnObject(fn._type);
	}

	public AggregateOperationTypes getAggOpType() {
		return _type;
	}

	/**
	 * Special case for weights w2==1
	 */
	@Override
	public Data execute(Data in1, double in2) {
		CmCovObject cm1=(CmCovObject) in1;
		
		if(cm1.isCMAllZeros()) {
			cm1.w=1;
			cm1.mean.set(in2, 0);
			cm1.min = in2;
			cm1.max = in2;
			cm1.m2.set(0,0);
			cm1.m3.set(0,0);
			cm1.m4.set(0,0);
			return cm1;
		}
		
		switch( _type )
		{
			case COUNT:
			{
				cm1.w = cm1.w + 1;
				break;
			}
			case MEAN:
			{
				double w= cm1.w + 1;
				double d=in2-cm1.mean._sum;
				cm1.mean=(KahanObject) _plus.execute(cm1.mean, d/w);
				cm1.w=w;
				break;
			}
			case MIN:
			{
				cm1.min = Math.min(cm1.min, in2);
				break;
			}
			case MAX:
			{
				cm1.max = Math.max(cm1.max, in2);
				break;
			}
			case CM2:
			{
				double w= cm1.w + 1;
				double d=in2-cm1.mean._sum;
				cm1.mean=(KahanObject) _plus.execute(cm1.mean, d/w);
				double t1=cm1.w/w*d;
				double lt1=t1*d;
				_buff2.set(cm1.m2);
				_buff2=(KahanObject) _plus.execute(_buff2, lt1);
				cm1.m2.set(_buff2);
				cm1.w=w;
				break;
			}
			case CM3:
			{
				double w = cm1.w + 1;
				double d=in2-cm1.mean._sum;
				cm1.mean=(KahanObject) _plus.execute(cm1.mean, d/w);
				double t1=cm1.w/w*d;
				double t2=-1/cm1.w;
				double lt1=t1*d;
				double lt2=Math.pow(t1, 3)*(1.0-Math.pow(t2, 2));
				double f2=1.0/w;
				_buff2.set(cm1.m2);
				_buff2=(KahanObject) _plus.execute(_buff2, lt1);
				_buff3.set(cm1.m3);
				_buff3=(KahanObject) _plus.execute(_buff3, lt2-3*cm1.m2._sum*f2*d);
				cm1.m2.set(_buff2);
				cm1.m3.set(_buff3);
				cm1.w=w;
				break;
			}
			case CM4:
			{
				double w=cm1.w+1;
				double d=in2-cm1.mean._sum;
				cm1.mean=(KahanObject) _plus.execute(cm1.mean, d/w);
				double t1=cm1.w/w*d;
				double t2=-1/cm1.w;
				double lt1=t1*d;
				double lt2=Math.pow(t1, 3)*(1.0-Math.pow(t2, 2));
				double lt3=Math.pow(t1, 4)*(1.0-Math.pow(t2, 3));
				double f2=1.0/w;
				_buff2.set(cm1.m2);
				_buff2=(KahanObject) _plus.execute(_buff2, lt1);
				_buff3.set(cm1.m3);
				_buff3=(KahanObject) _plus.execute(_buff3, lt2-3*cm1.m2._sum*f2*d);
				cm1.m4=(KahanObject) _plus.execute(cm1.m4, 6*cm1.m2._sum*Math.pow(-f2*d, 2) + lt3-4*cm1.m3._sum*f2*d);
				cm1.m2.set(_buff2);
				cm1.m3.set(_buff3);
				cm1.w=w;
				break;
			}
			case VARIANCE:
			{
				double w=cm1.w+1;
				double d=in2-cm1.mean._sum;
				cm1.mean=(KahanObject) _plus.execute(cm1.mean, d/w);
				double t1=cm1.w/w*d;
				double lt1=t1*d;
				cm1.m2=(KahanObject) _plus.execute(cm1.m2, lt1);
				cm1.w=w;
				break;
			}
			
			default:
				throw new DMLRuntimeException("Unsupported operation type: "+_type);
		}
		
		return cm1;
	}
	
	/**
	 * General case for arbitrary weights w2
	 */
	@Override
	public Data execute(Data in1, double in2, double w2) {
		CmCovObject cm1=(CmCovObject) in1;
		
		if(cm1.isCMAllZeros())
		{
			cm1.w=w2;
			cm1.mean.set(in2, 0);
			cm1.min = in2 * w2;
			cm1.max = in2 * w2;
			cm1.m2.set(0,0);
			cm1.m3.set(0,0);
			cm1.m4.set(0,0);
			return cm1;
		}
		
		switch( _type )
		{
			case COUNT:
			{
				cm1.w = Math.round(cm1.w + w2);
				break;
			}
			case MIN:
			{
				cm1.min = Math.min(cm1.min, in2 * w2);
				break;
			}
			case MAX:
			{
				cm1.max = Math.max(cm1.max, in2 * w2);
				break;
			}
			case MEAN:
			{
				double w = cm1.w + w2;
				double d=in2-cm1.mean._sum;
				cm1.mean=(KahanObject) _plus.execute(cm1.mean, w2*d/w);
				cm1.w=w;			
				break;
			}
			case CM2:
			{
				double w = cm1.w + w2;
				double d=in2-cm1.mean._sum;
				cm1.mean=(KahanObject) _plus.execute(cm1.mean, w2*d/w);
				double t1=cm1.w*w2/w*d;
				double lt1=t1*d;
				_buff2.set(cm1.m2);
				_buff2=(KahanObject) _plus.execute(_buff2, lt1);
				cm1.m2.set(_buff2);
				cm1.w=w;				
				break;
			}
			case CM3:
			{
				double w = cm1.w + w2;
				double d=in2-cm1.mean._sum;
				cm1.mean=(KahanObject) _plus.execute(cm1.mean, w2*d/w);
				double t1=cm1.w*w2/w*d;
				double t2=-1/cm1.w;
				double lt1=t1*d;
				double lt2=Math.pow(t1, 3)*(1/Math.pow(w2, 2)-Math.pow(t2, 2));
				double f2=w2/w;
				_buff2.set(cm1.m2);
				_buff2=(KahanObject) _plus.execute(_buff2, lt1);
				_buff3.set(cm1.m3);
				_buff3=(KahanObject) _plus.execute(_buff3, lt2-3*cm1.m2._sum*f2*d);
				cm1.m2.set(_buff2);
				cm1.m3.set(_buff3);
				cm1.w=w;
				break;
			}
			case CM4:
			{
				double w = cm1.w + w2;
				double d=in2-cm1.mean._sum;
				cm1.mean=(KahanObject) _plus.execute(cm1.mean, w2*d/w);
				double t1=cm1.w*w2/w*d;
				double t2=-1/cm1.w;
				double lt1=t1*d;
				double lt2=Math.pow(t1, 3)*(1/Math.pow(w2, 2)-Math.pow(t2, 2));
				double lt3=Math.pow(t1, 4)*(1/Math.pow(w2, 3)-Math.pow(t2, 3));
				double f2=w2/w;
				_buff2.set(cm1.m2);
				_buff2=(KahanObject) _plus.execute(_buff2, lt1);
				_buff3.set(cm1.m3);
				_buff3=(KahanObject) _plus.execute(_buff3, lt2-3*cm1.m2._sum*f2*d);
				cm1.m4=(KahanObject) _plus.execute(cm1.m4, 6*cm1.m2._sum*Math.pow(-f2*d, 2) + lt3-4*cm1.m3._sum*f2*d);
				cm1.m2.set(_buff2);
				cm1.m3.set(_buff3);
				cm1.w=w;
				break;
			}
			case VARIANCE:
			{
				double w = cm1.w + w2;
				double d=in2-cm1.mean._sum;
				cm1.mean=(KahanObject) _plus.execute(cm1.mean, w2*d/w);
				double t1=cm1.w*w2/w*d;
				double lt1=t1*d;
				cm1.m2=(KahanObject) _plus.execute(cm1.m2, lt1);
				cm1.w=w;
				break;
			}
			
			default:
				throw new DMLRuntimeException("Unsupported operation type: "+_type);
		}
		
		return cm1;
	}

	/**
	 * Combining stats from two partitions of the data.
	 */
	@Override
	public Data execute(Data in1, Data in2)
	{
		CmCovObject cm1=(CmCovObject) in1;
		CmCovObject cm2=(CmCovObject) in2;
		
		if(cm1.isCMAllZeros())
		{
			cm1.w=cm2.w;
			cm1.mean.set(cm2.mean);
			cm1.min = cm2.min;
			cm1.max = cm2.max;
			cm1.m2.set(cm2.m2);
			cm1.m3.set(cm2.m3);
			cm1.m4.set(cm2.m4);
			return cm1;
		}
		if(cm2.isCMAllZeros())
			return cm1;
		
		switch( _type )
		{
			case COUNT:
			{
				cm1.w = Math.round(cm1.w + cm2.w);				
				break;
			}
			case MIN:
			{
				cm1.min = Math.min(cm1.min, cm2.min);
				break;
			}
			case MAX:
			{
				cm1.max = Math.max(cm1.max, cm2.max);
				break;
			}
			case MEAN:
			{
				double w = cm1.w + cm2.w;
				double d=cm2.mean._sum-cm1.mean._sum;
				cm1.mean=(KahanObject) _plus.execute(cm1.mean, cm2.w*d/w);
				cm1.w=w;
				break;
			}
			case CM2:
			{
				double w = cm1.w + cm2.w;
				double d=cm2.mean._sum-cm1.mean._sum;
				cm1.mean=(KahanObject) _plus.execute(cm1.mean, cm2.w*d/w);
				double t1=cm1.w*cm2.w/w*d;
				double lt1=t1*d;
				_buff2.set(cm1.m2);
				_buff2=(KahanObject) _plus.execute(_buff2, cm2.m2._sum, cm2.m2._correction);
				_buff2=(KahanObject) _plus.execute(_buff2, lt1);
				cm1.m2.set(_buff2);
				cm1.w=w;
				break;
			}
			case CM3:
			{
				double w = cm1.w + cm2.w;
				double d=cm2.mean._sum-cm1.mean._sum;
				cm1.mean=(KahanObject) _plus.execute(cm1.mean, cm2.w*d/w);
				double t1=cm1.w*cm2.w/w*d;
				double t2=-1/cm1.w;
				double lt1=t1*d;
				double lt2=Math.pow(t1, 3)*(1/Math.pow(cm2.w, 2)-Math.pow(t2, 2));
				double f1=cm1.w/w;
				double f2=cm2.w/w;
				_buff2.set(cm1.m2);
				_buff2=(KahanObject) _plus.execute(_buff2, cm2.m2._sum, cm2.m2._correction);
				_buff2=(KahanObject) _plus.execute(_buff2, lt1);
				_buff3.set(cm1.m3);
				_buff3=(KahanObject) _plus.execute(_buff3, cm2.m3._sum, cm2.m3._correction);
				_buff3=(KahanObject) _plus.execute(_buff3, 3*(-f2*cm1.m2._sum+f1*cm2.m2._sum)*d + lt2);
				cm1.m2.set(_buff2);
				cm1.m3.set(_buff3);
				cm1.w=w;
				break;
			}
			case CM4:
			{
				double w = cm1.w + cm2.w;
				double d=cm2.mean._sum-cm1.mean._sum;
				cm1.mean=(KahanObject) _plus.execute(cm1.mean, cm2.w*d/w);
				double t1=cm1.w*cm2.w/w*d;
				double t2=-1/cm1.w;
				double lt1=t1*d;
				double lt2=Math.pow(t1, 3)*(1/Math.pow(cm2.w, 2)-Math.pow(t2, 2));
				double lt3=Math.pow(t1, 4)*(1/Math.pow(cm2.w, 3)-Math.pow(t2, 3));
				double f1=cm1.w/w;
				double f2=cm2.w/w;
				_buff2.set(cm1.m2);
				_buff2=(KahanObject) _plus.execute(_buff2, cm2.m2._sum, cm2.m2._correction);
				_buff2=(KahanObject) _plus.execute(_buff2, lt1);
				_buff3.set(cm1.m3);
				_buff3=(KahanObject) _plus.execute(_buff3, cm2.m3._sum, cm2.m3._correction);
				_buff3=(KahanObject) _plus.execute(_buff3, 3*(-f2*cm1.m2._sum+f1*cm2.m2._sum)*d + lt2);
				cm1.m4=(KahanObject) _plus.execute(cm1.m4, cm2.m4._sum, cm2.m4._correction);
				cm1.m4=(KahanObject) _plus.execute(cm1.m4, 4*(-f2*cm1.m3._sum+f1*cm2.m3._sum)*d 
						+ 6*(Math.pow(-f2, 2)*cm1.m2._sum+Math.pow(f1, 2)*cm2.m2._sum)*Math.pow(d, 2) + lt3);				
				cm1.m2.set(_buff2);
				cm1.m3.set(_buff3);
				cm1.w=w;
				break;
			}
			case VARIANCE:
			{
				double w = cm1.w + cm2.w;
				double d=cm2.mean._sum-cm1.mean._sum;
				cm1.mean=(KahanObject) _plus.execute(cm1.mean, cm2.w*d/w);
				double t1=cm1.w*cm2.w/w*d;
				double lt1=t1*d;
				cm1.m2=(KahanObject) _plus.execute(cm1.m2, cm2.m2._sum, cm2.m2._correction);
				cm1.m2=(KahanObject) _plus.execute(cm1.m2, lt1);
				cm1.w=w;
				break;
			}
			
			default:
				throw new DMLRuntimeException("Unsupported operation type: "+_type);
		}
		
		return cm1;
	}
}
