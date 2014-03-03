/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.functionobjects;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.KahanObject;


public class Mean extends ValueFunction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static Mean singleObj = null;
	
	private KahanPlus _plus = null; 
	
	private Mean() {
		_plus = KahanPlus.getKahanPlusFnObject();
	}
	
	public static Mean getMeanFnObject() {
		if ( singleObj == null )
			singleObj = new Mean();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	//overwride in1
	public Data execute(Data in1, double in2, double count) throws DMLRuntimeException {
		KahanObject kahanObj=(KahanObject)in1;
		double delta = (in2-kahanObj._sum)/count;
		_plus.execute(in1, delta);	
		return kahanObj;
	}
	
	/**
	 * Simplified version of execute(Data in1, double in2) 
	 * without exception handling and casts.
	 * 
	 * @param in1
	 * @param in2
	 */
	public void execute2(KahanObject in1, double in2, double count) 
	{
		double delta = (in2-in1._sum)/count;
		_plus.execute2(in1, delta);
	}
}
