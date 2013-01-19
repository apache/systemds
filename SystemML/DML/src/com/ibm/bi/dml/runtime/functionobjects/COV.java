package com.ibm.bi.dml.runtime.functionobjects;

import com.ibm.bi.dml.runtime.instructions.CPInstructions.CM_COV_Object;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.KahanObject;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class COV extends ValueFunction{

	private KahanPlus _plus = null; 
	
	public static COV getCOMFnObject() {
		//return new obj, required for correctness in multi-threaded
		//execution due to state in cm object
		return new COV(); 
	}
	
	private COV()
	{
		_plus = KahanPlus.getKahanPlusFnObject();
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	public Data execute(Data in1, double u, double v, double w2) throws DMLRuntimeException 
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
