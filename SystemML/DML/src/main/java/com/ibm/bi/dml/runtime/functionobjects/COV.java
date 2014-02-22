/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.functionobjects;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CM_COV_Object;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.KahanObject;


public class COV extends ValueFunction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static COV singleObj = null;
	
	private KahanPlus _plus = null; 
	
	public static COV getCOMFnObject() {
		if ( singleObj == null )
			singleObj = new COV();
		return singleObj;
	}
	
	private COV()
	{
		_plus = KahanPlus.getKahanPlusFnObject();
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	/**
	 * General case for arbitrary weights w2
	 * 
	 * @param in1
	 * @param u
	 * @param v
	 * @param w2
	 * @return
	 * @throws DMLRuntimeException
	 */
	public Data execute(Data in1, double u, double v, double w2) 
		throws DMLRuntimeException 
	{
		CM_COV_Object cov1=(CM_COV_Object) in1;
		if(cov1.isCOVAllZeros())
		{
			cov1.w=w2;
			cov1.mean.set(u, 0);
			cov1.mean_v.set(v, 0);
			cov1.c2.set(0,0);
			return cov1;
		}
		
		double w=(long)cov1.w+(long)w2;
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
	 */
	public Data execute(Data in1, double u, double v) 
		throws DMLRuntimeException 
	{
		CM_COV_Object cov1=(CM_COV_Object) in1;
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
	
	public Data execute(Data in1, Data in2) throws DMLRuntimeException 
	{
		CM_COV_Object cov1=(CM_COV_Object) in1;
		CM_COV_Object cov2=(CM_COV_Object) in2;
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
		
		double w=(long)cov1.w+(long)cov2.w;
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
