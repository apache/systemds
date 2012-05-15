package com.ibm.bi.dml.runtime.functionobjects;

import com.ibm.bi.dml.runtime.instructions.CPInstructions.CM_COV_Object;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.KahanObject;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class CM extends ValueFunction {

	private static CM singleObj = null;
	private static KahanPlus plus=KahanPlus.getKahanPlusFnObject();
	KahanObject buff2=new KahanObject(0, 0);
	KahanObject buff3=new KahanObject(0, 0);
	
	private CM() {
		// nothing to do here
	}
	
	public static CM getCMFnObject() {
		return singleObj = new CM(); //changed for multi-threaded exec  
		// if ( singleObj == null ) 
		//	return singleObj = new CM();
		//return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	//overwride in1
	public Data execute(Data in1, double in2, double w2) throws DMLRuntimeException {
		CM_COV_Object cm1=(CM_COV_Object) in1;
		if(cm1.isCMAllZeros())
		{
			cm1.w=w2;
			cm1.mean.set(in2, 0);
			cm1.m2.set(0,0);
			cm1.m3.set(0,0);
			cm1.m4.set(0,0);
			return cm1;
		}
		double w=(long)cm1.w+(long)w2;
		double d=in2-cm1.mean._sum;
		cm1.mean=(KahanObject) plus.execute(cm1.mean, w2*d/w);
		double t1=cm1.w*w2/w*d;
		double t2=-1/cm1.w;
		double lt1=t1*d;
		double lt2=Math.pow(t1, 3)*(1/Math.pow(w2, 2)-Math.pow(t2, 2));
		double lt3=Math.pow(t1, 4)*(1/Math.pow(w2, 3)-Math.pow(t2, 3));
		double f1=cm1.w/w;
		double f2=w2/w;
	//	double m2=cm1.m2 + lt1;
	//	double m3=cm1.m3 - 3*cm1.m2*f2*d + lt2;
	//	double m4=cm1.m4 - 4*cm1.m3*f2*d + 6*cm1.m2*Math.pow(-f2*d, 2) + lt3;
		buff2.set(cm1.m2);
		buff2=(KahanObject) plus.execute(buff2, lt1);
		buff3.set(cm1.m3);
		buff3=(KahanObject) plus.execute(buff3, lt2-3*cm1.m2._sum*f2*d);
		cm1.m4=(KahanObject) plus.execute(cm1.m4, 6*cm1.m2._sum*Math.pow(-f2*d, 2) + lt3-4*cm1.m3._sum*f2*d);
	//	cm1.mean=mean;
	//	cm1.mean_correction=kahan._correction;
		cm1.m2.set(buff2);
		cm1.m3.set(buff3);
		cm1.w=w;
		return cm1;
	}
	
	/*
	 //following the SPSS definition.
	public Data execute(Data in1, double in2, double w2) throws DMLRuntimeException {
		CMObject cm=(CMObject) in1;
		double oldweight=cm._weight;
		cm._weight+=w2;
		double v=w2/cm._weight*(in2-cm._mean);
		cm._mean+=v;
		double oldm2=cm._m2;
		double oldm3=cm._m3;
		double oldm4=cm._m4;
		double weightProduct=cm._weight*oldweight;
		double vsquare=Math.pow(v, 2);
		cm._m2=oldm2+weightProduct/w2*vsquare;
		cm._m3=oldm3-3*v*oldm2+weightProduct/Math.pow(w2,2)*(cm._weight-2*w2)*Math.pow(v, 3);
		cm._m4=oldm4-4*v*oldm3+6*vsquare*oldm2
		+((Math.pow(cm._weight, 2)-3*w2*oldweight)/Math.pow(w2,3))*Math.pow(v, 4)*weightProduct;
		return cm;
	}*/

	//overwride in1
	public Data execute(Data in1, Data in2) throws DMLRuntimeException 
	{
		CM_COV_Object cm1=(CM_COV_Object) in1;
		CM_COV_Object cm2=(CM_COV_Object) in2;
		if(cm1.isCMAllZeros())
		{
			cm1.w=cm2.w;
			cm1.mean.set(cm2.mean);
			cm1.m2.set(cm2.m2);
			cm1.m3.set(cm2.m3);
			cm1.m4.set(cm2.m4);
			return cm1;
		}
		
		if(cm2.isCMAllZeros())
			return cm1;
		
		double w=(long)cm1.w+(long)cm2.w;
		double d=cm2.mean._sum-cm1.mean._sum;
		cm1.mean=(KahanObject) plus.execute(cm1.mean, cm2.w*d/w);
		//double mean=cm1.mean+cm2.w*d/w;
		double t1=cm1.w*cm2.w/w*d;
		double t2=-1/cm1.w;
		double lt1=t1*d;
		double lt2=Math.pow(t1, 3)*(1/Math.pow(cm2.w, 2)-Math.pow(t2, 2));
		double lt3=Math.pow(t1, 4)*(1/Math.pow(cm2.w, 3)-Math.pow(t2, 3));
		double f1=cm1.w/w;
		double f2=cm2.w/w;
		buff2.set(cm1.m2);
		buff2=(KahanObject) plus.execute(buff2, cm2.m2._sum, cm2.m2._correction);
		buff2=(KahanObject) plus.execute(buff2, lt1);
		buff3.set(cm1.m3);
		buff3=(KahanObject) plus.execute(buff3, cm2.m3._sum, cm2.m3._correction);
		buff3=(KahanObject) plus.execute(buff3, 3*(-f2*cm1.m2._sum+f1*cm2.m2._sum)*d + lt2);
		cm1.m4=(KahanObject) plus.execute(cm1.m4, cm2.m4._sum, cm2.m4._correction);
		cm1.m4=(KahanObject) plus.execute(cm1.m4, 4*(-f2*cm1.m3._sum+f1*cm2.m3._sum)*d 
				+ 6*(Math.pow(-f2, 2)*cm1.m2._sum+Math.pow(f1, 2)*cm2.m2._sum)*Math.pow(d, 2) + lt3);
		
	/*	double m2=cm1.m2+cm2.m2 + lt1;
		double m3=cm1.m3+cm2.m3 + 3*(-f2*cm1.m2+f1*cm2.m2)*d + lt2;
		double m4=cm1.m4+cm2.m4 + 4*(-f2*cm1.m3+f1*cm2.m3)*d 
		+ 6*(Math.pow(-f2, 2)*cm1.m2+Math.pow(f1, 2)*cm2.m2)*Math.pow(d, 2) + lt3;*/
		cm1.m2.set(buff2);
		cm1.m3.set(buff3);
		cm1.w=w;
		return cm1;
	}
	/*
	 private double Q(CMObject cm1, CMObject cm2, int power)
	{
		return cm1._weight*Math.pow(cm1._mean,power)+cm2._weight*Math.pow(cm2._mean,power);
	}
	
	//following the SPSS definition, it is wrong
	public Data execute(Data in1, Data in2) throws DMLRuntimeException 
	{
		CMObject cm1=(CMObject) in1;
		CMObject cm2=(CMObject) in2;
		double w=cm1._weight+cm2._weight;
		double q1=cm1._mean*cm1._weight+cm2._mean*cm2._weight;
		double mean=q1/w;
		double p1=mean-cm1._mean;
		double p2=mean-cm2._mean;
		double q2=Q(cm1, cm2, 2);
		double q3=Q(cm1, cm2, 3);
		double q4=Q(cm1, cm2, 4);
		double mean2=Math.pow(mean, 2);
		double mean3=Math.pow(mean, 3);
		double mean4=Math.pow(mean, 4);
		double m2 = cm1._m2+cm2._m2 + q2 - 2*mean*q1 + w*mean2;
		double m3 = cm1._m3+cm2._m3 - 3*(p1*cm1._m2+p2*cm2._m2) 
		- 3*mean*(Math.pow(cm1._mean, 2)+Math.pow(cm2._mean, 2)) + 4*q3 - w*mean3;
		double m4 = cm1._m4+cm2._m4 - 4*(p1*cm1._m3+p2*cm2._m3) + 6*(Math.pow(p1, 2)*cm1._m2+Math.pow(p2, 2)*cm2._m2)-4*q4-4*mean*q3+6*mean2*q2-4*mean3*q1+2*w*mean4;
		cm1._m2=m2;
		cm1._m3=m3;
		cm1._m4=m4;
		cm1._mean=mean;
		cm1._weight=w;
		return cm1;
	}*/
}
