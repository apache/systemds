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

import java.util.HashMap;
import java.util.Map;

public class KeyTrieNode {
	private final Map<String, KeyTrieNode> children;
	private String key;
	private boolean check;
	private int count;

	public KeyTrieNode() {
		this.children = new HashMap<>();
		this.check = false;
		this.count = 1;
	}

	public KeyTrieNode(String key) {
		this.children = new HashMap<>();
		this.key = key;
		this.check = false;
		this.count = 1;
	}

	public void countPP() {
		this.count++;
	}

	public Map<String, KeyTrieNode> getChildren() {
		return children;
	}

	public String getKey() {
		return key;
	}

	public void setKey(String key) {
		this.key = key;
	}

	public boolean isCheck() {
		return check;
	}

	public void setCheck(boolean check) {
		this.check = check;
	}

	public int getCount() {
		return count;
	}

	public void setCount(int count) {
		this.count = count;
	}
}
