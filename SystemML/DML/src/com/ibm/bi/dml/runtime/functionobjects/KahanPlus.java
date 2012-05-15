package com.ibm.bi.dml.runtime.functionobjects;

import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.KahanObject;
import com.ibm.bi.dml.utils.DMLRuntimeException;


// Singleton class

public class KahanPlus extends ValueFunction {

	private static KahanPlus singleObj = null;
	
	private KahanPlus() {
		// nothing to do here
	}
	
	public static KahanPlus getKahanPlusFnObject() {
		return singleObj = new KahanPlus(); //changed for multi-threaded exec
		// if ( singleObj == null ) 
		//	return singleObj = new KahanPlus();
		//return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	//overwride in1
	public Data execute(Data in1, double in2) throws DMLRuntimeException {
		KahanObject kahanObj=(KahanObject)in1;
		double correction=in2+kahanObj._correction;
		double sum=kahanObj._sum+correction;
		kahanObj._correction=correction-(sum-kahanObj._sum);
		kahanObj._sum=sum;
		return kahanObj;
	}
	
	//overwride in1, in2 is the sum, in3 is the correction
	public Data execute(Data in1, double in2, double in3) throws DMLRuntimeException {
		KahanObject kahanObj=(KahanObject)in1;
		double correction=in2+(kahanObj._correction+in3);
		double sum=kahanObj._sum+correction;
		kahanObj._correction=correction-(sum-kahanObj._sum);
		kahanObj._sum=sum;
		return kahanObj;
	}
}
