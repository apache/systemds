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
		//kahanObj._correction=(correction-(sum-kahanObj._sum)); //MB: commented	
		//kahanObj._sum=sum;                                     //MB: commented
		kahanObj.set(sum, correction-(sum-kahanObj._sum));       //MB: instead of previous lines  
		
		/* TODO: Yuanyuan and Shirish, please review
		 * 
		 * Problem description: The tests ParForBivariateStatsTests produces
		 * (weakly reproducible) incorrect results (for COV). At the beginning
		 * iterations all results are correct and after a certain point 
		 * scale-scale computations return extremely small or extremely large values. 
		 * 
		 * Solution:  It seams that aggressive JVM optimizations in KahanPlus - targeting
		 * the algebraically 0 correction term - cause this problem. Those optimizations 
		 * are not applied in Debug mode and hence we didn't saw the error there. 
		 * Using a function call indirection for assigning sum and correction solved the 
		 * problem for many consecutive runs.
		 * 
		 * -- MB
		 * */
		
		return kahanObj;
	}
	
	//overwride in1, in2 is the sum, in3 is the correction
	public Data execute(Data in1, double in2, double in3) throws DMLRuntimeException {
		KahanObject kahanObj=(KahanObject)in1;
		double correction=in2+(kahanObj._correction+in3);
		double sum=kahanObj._sum+correction;
		//kahanObj._correction=correction-(sum-kahanObj._sum);
		//kahanObj._sum=sum;
		kahanObj.set(sum, correction-(sum-kahanObj._sum));
		//TODO: see method above
		
		return kahanObj;
	}
}
