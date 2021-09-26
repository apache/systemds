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

import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import java.util.*;

public class RawJSON {

	private String rawJSON;
	private ArrayList<String> rowJSON;

	public RawJSON(String rawJSON) {
		this.rawJSON = rawJSON.trim();
	}

	public void extractRows() {

		rowJSON = new ArrayList<>();
		Stack<Character> stack = new Stack<>();

		// remove "Enter" from input data
		rawJSON = rawJSON.replace("\n", "").replace("\r", "");
		StringBuilder row = new StringBuilder();
		for(Character ch : rawJSON.toCharArray()) {
			row.append(ch);
			if(ch.equals('{')) {
				stack.push(ch);
			}
			else if(ch.equals('}')) {
				stack.pop();
				if(stack.size() == 0) {
					rowJSON.add(row.toString());
					row = new StringBuilder();
				}
			}
		}
	}

	public void getL1Index() throws JSONException {
		for(String rs : rowJSON) {
			JSONObject jo = new JSONObject(rs);
			Map<String, Pair<Integer, Integer>> l0 = new HashMap<>();
			ArrayList<Object> rowObject = new ArrayList<>();
			l1Index(jo, rowObject, "");
		}
	}

	private void l1Index(JSONObject jo, ArrayList<Object> rowObject, String rootKey) throws JSONException {
		rowObject.add("{}");
		for(Iterator it = jo.keys(); it.hasNext(); ) {
			String key = (String) it.next();
			Object value = jo.get(key);
			key = rootKey.equals("") ? key : rootKey + "." + key;
			rowObject.add(key);
			if(value instanceof JSONObject) {
				l1Index((JSONObject) value, rowObject, key);
			}
			else if(value instanceof JSONArray) {
				JSONArray jArray = (JSONArray) value;
				l1Index(jArray, rowObject, key);
			}
			else {
				rowObject.add(value);
			}
		}
	}

	private void l1Index(JSONArray ja, ArrayList<Object> rowObject, String rootKey) throws JSONException {
		rowObject.add("[]");
		if(ja != null) {
			for(int i = 0; i < ja.length(); i++) {
				Object jaItem = ja.get(i);
				if(jaItem instanceof JSONObject) {
					l1Index((JSONObject) jaItem, rowObject, rootKey);
				}
				else if(jaItem instanceof JSONArray) {
					l1Index((JSONArray) jaItem, rowObject, rootKey);
				}
				else {
					if(jaItem instanceof Integer)
						rowObject.add(ja.getInt(i));
					else if(jaItem instanceof Long)
						rowObject.add(ja.getLong(i));
					else if(jaItem instanceof Float)
						rowObject.add(ja.getDouble(i));
					else if(jaItem instanceof Boolean)
						rowObject.add(ja.getBoolean(i));
					else if(jaItem instanceof String)
						rowObject.add(ja.getString(i));
					else if(jaItem instanceof Short)
						rowObject.add(ja.getShort(i));
					else {
						throw new RuntimeException("Value type of the JSON item don't recognized!!");
					}
				}
			}
		}
	}
}
