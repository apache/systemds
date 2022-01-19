/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.iogen;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Stack;

public class FastJSONIndex {

	private final ArrayList<Object> l1Index;
	private final Map<String, Integer> l0Index;

	public FastJSONIndex(String input) throws JSONException {
		l1Index = new ArrayList<>();
		l0Index = new HashMap<>();
		JSONObject jo = new JSONObject(input);
		lIndex(jo, new Stack<>(), -1);
	}

	private void lIndex(JSONObject jo, Stack<String> keyChain, int index) throws JSONException {

		for(Iterator it = jo.keys(); it.hasNext(); ) {
			String key = (String) it.next();
			Object value = jo.get(key);
			keyChain.add(key);
			if(value instanceof JSONObject) {
				JSONObject jon = (JSONObject) value;
				lIndex(jon, keyChain, index);
			}
			else if(value instanceof JSONArray) {
				JSONArray ja = (JSONArray) value;
				lIndex(ja, keyChain);
			}
			else {
				l0Index.put(getKeysAsString(keyChain), l1Index.size());
				l1Index.add(value);
			}
			keyChain.pop();
		}
	}

	private void lIndex(JSONArray ja, Stack<String> keyChain) throws JSONException {
		if(ja != null) {
			for(int i = 0; i < ja.length(); i++) {

				Object value = ja.get(i);
				keyChain.add(i + "");
				if(value instanceof JSONObject) {
					lIndex((JSONObject) value, keyChain, i);
				}
				else if(value instanceof JSONArray) {
					lIndex((JSONArray) value, keyChain);
				}
				else {
					l0Index.put(getKeysAsString(keyChain), l1Index.size());
					l1Index.add(value);
				}
				keyChain.pop();
			}
		}
	}

	public String getKeysAsString(Stack<String> keyChain) {

		StringBuilder s = new StringBuilder();
		for(String k : keyChain)
			s.append(k).append(".");
		s.deleteCharAt(s.length() - 1);
		return s.toString();
	}

	public ArrayList<Object> getL1Index() {
		return l1Index;
	}

	public Map<String, Integer> getL0Index() {
		return l0Index;
	}

	public double getDoubleValue(String key) {
		Integer index = l0Index.get(key);
		if(index == null)
			return 0;
		else
			return UtilFunctions.getDouble(l1Index.get(index));
	}

	public Object getObjectValue(String key) {
		Integer index = l0Index.get(key);
		if(index == null)
			return null;
		else
			return l1Index.get(index);
	}

	public ArrayList<String> getNames() {
		ArrayList<String> result = new ArrayList<>();
		for(String k : l0Index.keySet())
			result.add(k);
		return result;
	}

	public Map<String, Types.ValueType> getNamesType() {
		Map<String, Types.ValueType> result = new HashMap<>();
		for(String k : l0Index.keySet()) {
			Object o = l1Index.get(l0Index.get(k));
			Types.ValueType vt = getValueType(o);
			result.put(k, vt);
		}
		return result;
	}

	private Types.ValueType getValueType(Object o) {
		Types.ValueType vt;
		if(o instanceof Integer)
			vt = Types.ValueType.INT32;
		else if(o instanceof Long)
			vt = Types.ValueType.INT64;
		else if(o instanceof Float)
			vt = Types.ValueType.FP32;
		else if(o instanceof Double)
			vt = Types.ValueType.FP64;
		else if(o instanceof String)
			vt = Types.ValueType.STRING;
		else if(o instanceof Boolean)
			vt = Types.ValueType.BOOLEAN;
		else
			throw new RuntimeException("Don't support object value type!");
		return vt;
	}
}
