package com.ibm.bi.dml.api;

import com.ibm.jaql.json.type.JsonBool;
import com.ibm.jaql.json.type.JsonString;
import com.ibm.bi.dml.utils.AppException;



/** 
 * JAQL Java UDF that will be used by DML app to throw exception
 * including data transformation, post-processing etc.
 * 
 */

public class AppExceptionJaqlUdf {
	
	public static JsonBool eval(JsonString errMsg) throws AppException {

		if (errMsg.length() > 0) {
		  throw new AppException(errMsg.toString());
		}
		return JsonBool.FALSE;
	}

	public static JsonBool eval(JsonString errMsg, Exception e)
			throws AppException {
		if (errMsg.length() > 0) {
		   throw new AppException(errMsg.toString(), e);
		}
		return JsonBool.FALSE;
	}

}
