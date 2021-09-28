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

import java.util.ArrayList;
import java.util.Stack;

public class JSONIndexProperties {

	public enum JSONItemType {
		PRIMITIVE, ARRAY, OBJECT;

		@Override public String toString() {
			return this.name().toLowerCase();
		}
	}

	private Types.ValueType valueType;
	private final ArrayList<String> keys;
	private final JSONItemType itemType;
	private final int index;
	private final int size;

	public JSONIndexProperties(Stack<String> keys, JSONItemType itemType, int size, int index) {
		this.keys = new ArrayList<>();
		this.keys.addAll(keys);
		this.itemType = itemType;
		this.size = size;
		this.index = index;
	}

	public Types.ValueType getValueType() {
		return valueType;
	}

	public void setValueType(Types.ValueType valueType) {
		this.valueType = valueType;
	}

	public int getIndex() {
		return index;
	}

	public ArrayList<String> getKeys() {
		return keys;
	}

	public JSONItemType getItemType() {
		return itemType;
	}

	public int getSize() {
		return size;
	}

	public String getKeysAsString() {

		StringBuilder s = new StringBuilder();
		for(String k : keys)
			s.append(k).append(".");
		s.deleteCharAt(s.length() - 1);
		return s.toString();
	}
}
