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

import com.google.gson.Gson;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class RawRowJSON {
	private final String raw;
	private final ArrayList<Object> l1Index;
	private final Map<String, IndexProperties> l0Index;
	private final BitSet reserved;
	private int lastIndex;

	public enum JSONItemType {
		PRIMITIVE,
		ARRAY,
		OBJECT;
		@Override
		public String toString() {
			return this.name().toLowerCase();
		}
	}

	public RawRowJSON(String raw) {
		this.raw = raw;
		l1Index = new ArrayList<>();
		l0Index = new HashMap<>();

		try {
			JSONObject jo = new JSONObject(raw);
			lIndex(jo, l1Index, l0Index, "");
			reserved = new BitSet(l1Index.size());
			print();
		}
		catch(JSONException e){
			throw new RuntimeException(e);
		}
	}

	/* Index JSON values. The index have two levels.
	The first level reconstruct the json text format, and the second level
	index they keys in json string.
	*/
	private void lIndex(JSONObject jo, ArrayList<Object> l1, Map<String, IndexProperties> l0, String rootKey)
		throws JSONException {
		for(Iterator it = jo.keys(); it.hasNext(); ) {
			String key = (String) it.next();
			Object value = jo.get(key);
			key = rootKey.equals("") ? key : rootKey + "." + key;
			JSONItemType jit;
			if(value instanceof JSONObject)
				jit = JSONItemType.OBJECT;
			else if(value instanceof JSONArray)
				jit = JSONItemType.ARRAY;
			else
				jit = JSONItemType.PRIMITIVE;

			IndexProperties ip = new IndexProperties(jit, l1.size());
			switch(jit){
				case OBJECT:
					lIndex((JSONObject) value, l1, l0, key);
					break;

				case ARRAY:
					JSONArray jArray = (JSONArray) value;
					lIndex(jArray, l1, l0, key);
					break;
				case PRIMITIVE:
					l1.add(value);
					break;
			}
			ip.setSize(l1.size() - ip.getIndex());
			l0.put(key, ip);
		}
	}

	private void lIndex(JSONArray ja, ArrayList<Object> l1, Map<String, IndexProperties> l0, String rootKey)
		throws JSONException {
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
					l1Index.add(jaItem);
				}
			}
		}
	}


	public Pair<String, Integer> findValue(Object value, Types.ValueType vt, boolean update) {
		for(Object o: l1Index){

		}
		return null;
	}

	public void print(){
		Gson gson=new Gson();
		System.out.println("-----------------------------------------------------");
		System.out.println("RAW: "+ raw);
		System.out.println(gson.toJson(l1Index));
		System.out.println(gson.toJson(l0Index));
		System.out.println("-----------------------------------------------------");
	}

	private class IndexProperties{
		private final JSONItemType type;
		private final int index;
		private int size;

		public IndexProperties(JSONItemType type, int index) {
			this.type = type;
			this.index = index;
		}

		public JSONItemType getType() {
			return type;
		}

		public int getIndex() {
			return index;
		}

		public void setSize(int size) {
			this.size = size;
		}

		public int getSize() {
			return size;
		}
	}
}
