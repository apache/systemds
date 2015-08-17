/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.utils;

import java.io.IOException;
import java.io.Reader;

import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

public class JSONHelper {
	public static Object get(JSONObject jsonObject, String key) {
		Object result = null;
		try {
			if(jsonObject != null) {
				result = jsonObject.get(key);
			}
		} catch (JSONException e) {
			// ignore and return null
		}
		
		return result;
	}
	
	public static JSONObject parse(Reader reader) throws IOException {
		try { 
			if(reader != null) {
				JSONObject result = new JSONObject(reader); 
				return result;
			} else {
				return null;
			}
			
		} catch (JSONException je) {
			throw new IOException("Error parsing json", je);
		}
    }
}
