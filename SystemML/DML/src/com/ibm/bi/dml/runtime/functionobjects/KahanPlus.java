/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.functionobjects;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.KahanObject;


public class KahanPlus extends ValueFunction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private KahanPlus() {
		// nothing to do here
	}
	
	public static KahanPlus getKahanPlusFnObject() {
		//return new obj, required for correctness in multi-threaded
		//execution due to state in kahan object
		return new KahanPlus();
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
		//kahanObj._correction=(correction-(sum-kahanObj._sum)); 	
		//kahanObj._sum=sum;                                     
		kahanObj.set(sum, correction-(sum-kahanObj._sum)); //prevent eager JIT opt 		
		return kahanObj;
	}
	
	//overwride in1, in2 is the sum, in3 is the correction
	public Data execute(Data in1, double in2, double in3) throws DMLRuntimeException {
		KahanObject kahanObj=(KahanObject)in1;
		double correction=in2+(kahanObj._correction+in3);
		double sum=kahanObj._sum+correction;
		//kahanObj._correction=correction-(sum-kahanObj._sum);
		//kahanObj._sum=sum;
		kahanObj.set(sum, correction-(sum-kahanObj._sum)); //prevent eager JIT opt
		return kahanObj;
	}
}
