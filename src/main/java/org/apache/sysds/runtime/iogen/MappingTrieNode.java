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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class MappingTrieNode {

	public enum Type {
		INNER, END, IGNORE;
		@Override
		public String toString() {
			return this.name().toUpperCase();
		}
	}

	private Map<String, MappingTrieNode> children;
	private Type nodeType;
	private ArrayList<Integer> rowIndexes;

	public MappingTrieNode(Type nodeType) {
		this.nodeType = nodeType;
		children = new HashMap<>();
		rowIndexes = new ArrayList<>();
	}

	public MappingTrieNode() {
		this.nodeType = Type.END;
		children = new HashMap<>();
		rowIndexes = new ArrayList<>();
	}

	public Map<String, MappingTrieNode> getChildren() {
		return children;
	}

	public void setChildren(Map<String, MappingTrieNode> children) {
		this.children = children;
	}

	public Type getNodeType() {
		return nodeType;
	}

	public void setNodeType(Type nodeType) {
		this.nodeType = nodeType;
	}

	public void addRowIndex(int rowIndex) {
		rowIndexes.add(rowIndex);
	}

	public void addRowIndex(ArrayList<Integer> rowIndexes) {
		this.rowIndexes.addAll(rowIndexes);
	}

	public void setRowIndexes(ArrayList<Integer> rowIndexes) {
		this.rowIndexes = rowIndexes;
	}

	public ArrayList<Integer> getRowIndexes() {
		return rowIndexes;
	}
}
