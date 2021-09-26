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

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Stack;

public abstract class ReaderMappingJSON {
	protected String[] mapCol;
	protected String rawJSON;
	protected ArrayList<String> rowJSON;
	protected ArrayList<ArrayList<Object>> l1Index;
	protected ArrayList<Map<String,Integer>> l0Index;
	protected static int nrows;
	protected static int ncols;

	public ReaderMappingJSON(String raw) {
		rawJSON = raw.trim();
		l1Index = new ArrayList<>();
		l0Index = new ArrayList<>();
		try {
			extractRows();
			doLIndex();
		}
		catch(JSONException e){
			throw new RuntimeException(e);
		}
	}

	// Extract all rows from plain text raw data
	protected void extractRows() {
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

	/* Index JSON values. The index have two levels.
	The first level reconstruct the json text format, and the second level
	index they keys in json string.
	Example:
		sample json string: {"a":1, "b":"HELLO", "c":[1,2,3]}
		level 1(array): {} a 1 b HELLO c [] 1 2 3
		level 0(map)  : (a,1) (b,3) (c,5)
	*/
	protected void doLIndex() throws JSONException {
		for(String rs : rowJSON) {
			JSONObject jo = new JSONObject(rs);
			Map<String, Integer> l0 = new HashMap<>();
			ArrayList<Object> l1 = new ArrayList<>();
			lIndex(jo, l1, l0, "");
			l1Index.add(l1);
			l0Index.add(l0);
		}
	}

	protected void lIndex(JSONObject jo, ArrayList<Object> l1, Map<String, Integer> l0, String rootKey) throws JSONException {
		l1.add("{}");
		for(Iterator it = jo.keys(); it.hasNext(); ) {
			String key = (String) it.next();
			Object value = jo.get(key);
			key = rootKey.equals("") ? key : rootKey + "." + key;
			l0.put(key, l1.size());
			l1.add(key);
			if(value instanceof JSONObject) {
				lIndex((JSONObject) value, l1, l0, key);
			}
			else if(value instanceof JSONArray) {
				JSONArray jArray = (JSONArray) value;
				lIndex(jArray, l1, l0, key);
			}
			else {
				l1.add(value);
			}
		}
	}

	protected void lIndex(JSONArray ja, ArrayList<Object> l1, Map<String, Integer> l0, String rootKey) throws JSONException {
		l1.add("[]");
		if(ja != null) {
			for(int i = 0; i < ja.length(); i++) {
				Object jaItem = ja.get(i);
				if(jaItem instanceof JSONObject) {
					lIndex((JSONObject) jaItem, l1, l0, rootKey);
				}
				else if(jaItem instanceof JSONArray) {
					lIndex((JSONArray) jaItem, l1, l0, rootKey);
				}
				else {
					if(jaItem instanceof Integer)
						l1.add(ja.getInt(i));
					else if(jaItem instanceof Long)
						l1.add(ja.getLong(i));
					else if(jaItem instanceof Float)
						l1.add(ja.getDouble(i));
					else if(jaItem instanceof Boolean)
						l1.add(ja.getBoolean(i));
					else if(jaItem instanceof String)
						l1.add(ja.getString(i));
					else if(jaItem instanceof Short)
						l1.add(ja.getShort(i));
					else {
						throw new RuntimeException("Value type of the JSON item don't recognized!!");
					}
				}
			}
		}
	}

	// Matrix Reader Mapping
	public static class MatrixReaderMapping extends ReaderMappingJSON {

		private MatrixBlock sampleMatrix;

		public MatrixReaderMapping(String raw, MatrixBlock matrix){
			super(raw);
			this.sampleMatrix = matrix;
			nrows = sampleMatrix.getNumRows();
			ncols = sampleMatrix.getNumColumns();

		}
	}
}
