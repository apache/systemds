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

import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.KahanObject;


/**
 * GENERAL NOTE:
 * * 05/28/2014: We decided to do handle weights consistently to SPSS in an operation-specific manner, 
 *   i.e., we (1) round instead of casting where required (e.g. count), and (2) consistently use
 *   fractional weight values elsewhere. In case a count-base interpretation of weights is needed, just 
 *   ensure rounding before calling CM/COV/KahanPlus.
 * 
 */
public class COV extends ValueFunction
{
	private static final long serialVersionUID = 1865050401811477181L;

	private static COV singleObj = null;
	
	private KahanPlus _plus = null; 
	
	public static COV getCOMFnObject() {
		if ( singleObj == null )
			singleObj = new COV();
		return singleObj;
	}
	
	private COV() {
		_plus = KahanPlus.getKahanPlusFnObject();
	}
	
	/**
	 * General case for arbitrary weights w2
	 * 
	 * @param in1 input data
	 * @param u ?
	 * @param v ?
	 * @param w2 ?
	 * @return result
	 */
	@Override
	public Data execute(Data in1, double u, double v, double w2) 
	{
		CmCovObject cov1=(CmCovObject) in1;
		if(cov1.isCOVAllZeros())
		{
			cov1.w=w2;
			cov1.mean.set(u, 0);
			cov1.mean_v.set(v, 0);
			cov1.c2.set(0,0);
			return cov1;
		}
		
		double w = cov1.w + w2;
		double du=u-cov1.mean._sum;
		double dv=v-cov1.mean_v._sum;
		cov1.mean=(KahanObject) _plus.execute(cov1.mean, w2*du/w);
		cov1.mean_v=(KahanObject) _plus.execute(cov1.mean_v, w2*dv/w);
		cov1.c2=(KahanObject) _plus.execute(cov1.c2, cov1.w*w2/w*du*dv);
		cov1.w=w;
		
		return cov1;
	}
	
	/**
	 * Special case for weights w2==1
	 * 
	 * @param in1 ?
	 * @param u ?
	 * @param v ?
	 * @return result
	 */
	@Override
	public Data execute(Data in1, double u, double v) 
	{
		CmCovObject cov1=(CmCovObject) in1;
		if(cov1.isCOVAllZeros())
		{
			cov1.w=1L;
			cov1.mean.set(u, 0);
			cov1.mean_v.set(v, 0);
			cov1.c2.set(0,0);
			return cov1;
		}
		
		double w=cov1.w+1;
		double du=u-cov1.mean._sum;
		double dv=v-cov1.mean_v._sum;
		cov1.mean=(KahanObject) _plus.execute(cov1.mean, du/w);
		cov1.mean_v=(KahanObject) _plus.execute(cov1.mean_v, dv/w);
		cov1.c2=(KahanObject) _plus.execute(cov1.c2, cov1.w/w*du*dv);
		cov1.w=w;
		
		return cov1;
	}
	
	@Override
	public Data execute(Data in1, Data in2)
	{
		CmCovObject cov1=(CmCovObject) in1;
		CmCovObject cov2=(CmCovObject) in2;
		if(cov1.isCOVAllZeros())
		{
			cov1.w=cov2.w;
			cov1.mean.set(cov2.mean);
			cov1.mean_v.set(cov2.mean_v);
			cov1.c2.set(cov2.c2);
			return cov1;
		}
		
		if(cov2.isCOVAllZeros())
			return cov1;
		
		double w = cov1.w + cov2.w;
		double du=cov2.mean._sum-cov1.mean._sum;
		double dv=cov2.mean_v._sum-cov1.mean_v._sum;		
		cov1.mean=(KahanObject) _plus.execute(cov1.mean, cov2.w*du/w);
		cov1.mean_v=(KahanObject) _plus.execute(cov1.mean_v, cov2.w*dv/w);
		cov1.c2=(KahanObject) _plus.execute(cov1.c2, cov2.c2._sum, cov2.c2._correction);
		cov1.c2=(KahanObject) _plus.execute(cov1.c2, cov1.w*cov2.w/w*du*dv);		
		cov1.w=w;
		
		return cov1;
	}
}
