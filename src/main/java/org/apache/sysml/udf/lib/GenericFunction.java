package org.apache.sysml.udf.lib;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.lang.StringUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.udf.FunctionParameter;
import org.apache.sysml.udf.Matrix;
import org.apache.sysml.udf.PackageFunction;
import org.apache.sysml.udf.Scalar;
import scala.Function0;

public class GenericFunction extends PackageFunction {
	private static final long serialVersionUID = -195996547505886575L;
	String [] fnSignature;
	FunctionParameter [] returnVals;
	Function0<FunctionParameter []> scalaUDF; 
	
	public GenericFunction(String [] fnSignature) {
		this.fnSignature = fnSignature;
	}
	public void setScalaUDF(Function0<FunctionParameter []> scalaUDF) {
		this.scalaUDF = scalaUDF;
	}
	
	@Override
	public int getNumFunctionOutputs() {
		String retSignature = fnSignature[fnSignature.length -1];
		if(!retSignature.startsWith("("))
			return 1;
		else {
			return StringUtils.countMatches(retSignature, ",") + 1;
		}
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) {
		if(returnVals == null || returnVals.length <= pos)
			throw new RuntimeException("Incorrect number of outputs or function not executed");
		return returnVals[pos];
	}

	@Override
	public void execute() {
		returnVals = scalaUDF.apply();
	}
	
//	public Object constructOutput(Object ret) throws DMLRuntimeException, IOException {
//		if(ret instanceof Integer)
//			return new Scalar(ScalarValueType.Integer, String.valueOf(ret));
//		else if(ret instanceof Double)
//			return new Scalar(ScalarValueType.Double, String.valueOf(ret));
//		else if(ret instanceof String)
//			return new Scalar(ScalarValueType.Text, String.valueOf(ret));
//		else if(ret instanceof Boolean)
//			return new Scalar(ScalarValueType.Boolean, String.valueOf(ret));
//		else if(ret instanceof Boolean)
//			
//	}
	
	public Object getInput(String type, int pos) throws DMLRuntimeException, IOException {
		if(type.equals("Int") || type.equals("java.lang.Integer")) {
			return Integer.parseInt(((Scalar)getFunctionInput(pos)).getValue());
		}
		else if(type.equals("Double") || type.equals("java.lang.Double")) {
			return Double.parseDouble(((Scalar)getFunctionInput(pos)).getValue());
		}
		else if(type.equals("java.lang.String")) {
			return ((Scalar)getFunctionInput(pos)).getValue();
		}
		else if(type.equals("boolean") || type.equals("java.lang.Boolean")) {
			return Boolean.parseBoolean(((Scalar)getFunctionInput(pos)).getValue());
		}
		else if(type.equals("scala.Array[scala.Array[Double]]")) {
			return ((Matrix) getFunctionInput(pos)).getMatrixAsDoubleArray();
		}
		
		throw new RuntimeException("Unsupported type: " + type);
	}

}
