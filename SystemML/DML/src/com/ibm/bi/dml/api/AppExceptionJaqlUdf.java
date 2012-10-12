package com.ibm.bi.dml.api;

import com.ibm.jaql.json.type.JsonString;
import com.ibm.bi.dml.utils.AppException;



/** 
 * JAQL Java UDF that will be used by DML app to throw exception
 * including data transformation, post-processing etc.
 * 
 */

public class AppExceptionJaqlUdf {
	
	public static void eval(JsonString errMsg) throws AppException {

		throw new AppException(errMsg.toString());
	}

	public static void eval(JsonString errMsg, Exception e)
			throws AppException {
		throw new AppException(errMsg.toString(), e);
	}

}
