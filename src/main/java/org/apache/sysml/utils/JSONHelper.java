/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.utils;

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
