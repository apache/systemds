package com.ibm.bi.dml.packagesupport;

//import com.ibm.bi.dml.packagesupport.Scalar;
//import com.ibm.bi.dml.packagesupport.FIO;

import com.ibm.bi.dml.packagesupport.Scalar.ScalarType;
import com.ibm.bi.dml.packagesupport.PackageFunction;
import com.ibm.bi.dml.packagesupport.PackageRuntimeException;

/**
 * Wrapper class for time invocation
 * 
 * time = externalFunction(Integer i) return (Double B) implemented in
 * (classname="com.ibm.bi.dml.packagesupport.TimeWrapper",exectype="mem");
 * 
 * t = time (1);
 * 
 * print( "Time in millsecs= " + t);
 * 
 */
public class TimeWrapper extends PackageFunction {
	private static final long serialVersionUID = 1L;

	private Scalar _ret;

	@Override
	public int getNumFunctionOutputs() {
		return 1;
	}

	@Override
	public FIO getFunctionOutput(int pos) {
		if (pos == 0)
			return _ret;

		throw new PackageRuntimeException(
				"Invalid function output being requested");
	}

	@Override
	public void execute() {
		try {
			// int past = Integer.parseInt(((Scalar) getFunctionInput(0))
			// .getValue());
			long present = System.currentTimeMillis();
			_ret = new Scalar(ScalarType.Double, String.valueOf(present));
		} catch (Exception e) {
			throw new PackageRuntimeException(
					"Error executing external time function", e);
		}
	}

}
