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

public class RawRowJSON {
	private final ArrayList<Object> l1Index;
	private final ArrayList<JSONIndexProperties> l0Index;
	private final Map<String, Integer> l0IndexMap;
	private final RawJSONObject rawObject;

	public RawRowJSON(String raw) {
		l1Index = new ArrayList<>();
		l0Index = new ArrayList<>();
		l0IndexMap = new HashMap<>();
		rawObject = new RawJSONObject("Root");
		try {
			JSONObject jo = new JSONObject(raw);
			lIndex(jo, new Stack<>(), -1, rawObject);
			for(int i = 0; i < l0Index.size(); i++) {
				l0IndexMap.put(l0Index.get(i).getKeysAsString(), i);
			}
		}
		catch(JSONException e) {
			throw new RuntimeException(e);
		}
	}

	/* Index JSON values. The index have two levels.
	The first level reconstruct the json text format, and the second level
	index they keys in json string.
	*/
	private void lIndex(JSONObject jo, Stack<String> keyChain, int index, RawJSONObject rawObject) throws JSONException {
		for(Iterator it = jo.keys(); it.hasNext(); ) {
			String key = (String) it.next();
			Object value = jo.get(key);
			keyChain.add(key);
			if(value instanceof JSONObject) {
				JSONObject jon = (JSONObject) value;
				RawJSONObject objectItem = new RawJSONObject(key);
				rawObject.addObjectItem(objectItem);
				lIndex(jon, keyChain, index, objectItem);
			}
			else if(value instanceof JSONArray) {
				JSONArray ja = (JSONArray) value;
				RawJSONObject arrayItem = new RawJSONObject(key);
				rawObject.addArrayItem(arrayItem);
				lIndex(ja, keyChain, arrayItem);
			}
			else {
				l1Index.add(value);
				l0Index.add(new JSONIndexProperties(keyChain, JSONIndexProperties.JSONItemType.PRIMITIVE, 1, index));
				rawObject.addPrimitiveItem(new RawJSONObject.PrimitiveItem(key, getValueType(value)));
			}
			keyChain.pop();
		}
	}

	private void lIndex(JSONArray ja, Stack<String> keyChain, RawJSONObject rawObject) throws JSONException {
		if(ja != null) {
			for(int i = 0; i < ja.length(); i++) {

				Object value = ja.get(i);
				keyChain.add(i + "");
				if(value instanceof JSONObject) {
					RawJSONObject objectItem = new RawJSONObject(i+"");
					rawObject.addObjectItem(objectItem);
					lIndex((JSONObject) value, keyChain, i, objectItem);
				}
				else if(value instanceof JSONArray) {
					RawJSONObject arrayItem = new RawJSONObject(i+"");
					rawObject.addArrayItem(arrayItem);
					lIndex((JSONArray) value, keyChain, arrayItem);
				}
				else {
					l1Index.add(value);
					l0Index.add(new JSONIndexProperties(keyChain, JSONIndexProperties.JSONItemType.PRIMITIVE, 1, i));
					Types.ValueType vt = getValueType(value);
					rawObject.addPrimitiveItem(new RawJSONObject.PrimitiveItem("i_"+i, vt));
				}
				keyChain.pop();
			}
		}
	}

	public ArrayList<String> getSchemaNames() {
		int size = l0Index.size();
		ArrayList<String> names = new ArrayList<>();
		for(int i = 0; i < size; i++) {
			JSONIndexProperties jip = l0Index.get(i);
			String key = jip.getKeysAsString();
			names.add(key);
		}
		return names;
	}

	public Map<String, Types.ValueType> getSchema() {
		int size = l0Index.size();
		Map<String, Types.ValueType> schema = new HashMap<>();

		for(int i = 0; i < size; i++) {
			JSONIndexProperties jip = l0Index.get(i);
			String key = jip.getKeysAsString();
			Object value = l1Index.get(i);
			schema.put(key, getValueType(value));
		}
		return schema;
	}

	private Types.ValueType getValueType(Object value){
		Types.ValueType vt;
		if(value instanceof Integer)
			vt = Types.ValueType.INT32;
		else if(value instanceof Long)
			vt = Types.ValueType.INT64;
		else if(value instanceof Float)
			vt = Types.ValueType.FP32;
		else if(value instanceof Double)
			vt = Types.ValueType.FP64;
		else if(value instanceof Boolean)
			vt = Types.ValueType.BOOLEAN;
		else if(value instanceof String)
			vt = Types.ValueType.STRING;
		else
			throw new RuntimeException("Can't recognize the value type of object!");
		return vt;
	}

	public double getDoubleValue(String key) {
		Integer index = l0IndexMap.get(key);
		if(index == null)
			return 0;
		else
			return UtilFunctions.getDouble(l1Index.get(index));
	}

	public Object getObjectValue(String key) {
		Integer index = l0IndexMap.get(key);
		if(index == null)
			return null;
		else
			return l1Index.get(index);

	}
}
