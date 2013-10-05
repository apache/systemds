/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.api;

import com.ibm.jaql.json.type.JsonBool;
import com.ibm.jaql.json.type.JsonString;
import com.ibm.bi.dml.utils.AppException;



/** 
 * JAQL Java UDF that will be used by DML app to throw exception
 * including data transformation, post-processing etc.
 * 
 */

public class AppExceptionJaqlUdf 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
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
