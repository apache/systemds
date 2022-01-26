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

package org.apache.sysds.runtime.iogen.codegen;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.iogen.CustomProperties;
import org.apache.sysds.runtime.iogen.KeyTrie;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

public class CodeGenTrie {
	private final CodeGenTrieNode rootCol;
	private final CodeGenTrieNode rootRow;
	private final CustomProperties properties;
	private final String destination;

	public CodeGenTrie(CustomProperties properties, String destination){
		this.rootCol = new CodeGenTrieNode(CodeGenTrieNode.NodeType.COL);
		this.rootRow = new CodeGenTrieNode(CodeGenTrieNode.NodeType.ROW);
		this.properties = properties;
		this.destination = destination;
		buildPrefixTree();
	}

	// Build Trie for Col and Row Key Patterns
	private void buildPrefixTree(){
		for(int c=0; c< properties.getColKeyPattern().length; c++){
			KeyTrie keyTrie = properties.getColKeyPattern()[c];
			Types.ValueType vt = properties.getSchema() == null ? Types.ValueType.FP64 : properties.getSchema()[c];
			for(ArrayList<String> keys : keyTrie.getReversePrefixKeyPatterns())
				this.insert(rootCol, c, vt, keys);
		}

		if(properties.getRowIndex() == CustomProperties.IndexProperties.PREFIX){
			KeyTrie keyTrie = properties.getRowKeyPattern();
			Types.ValueType vt = Types.ValueType.FP32;
			for(ArrayList<String> keys : keyTrie.getReversePrefixKeyPatterns())
				this.insert(rootRow, -1, vt, keys);
		}
	}

	private void insert(CodeGenTrieNode root ,int index, Types.ValueType valueType, ArrayList<String> keys) {
		CodeGenTrieNode currentNode = root;
		int rci = 0;
		for(String key : keys) {
			if(currentNode.getChildren().containsKey(key)) {
				currentNode = currentNode.getChildren().get(key);
				rci++;
			}
			else
				break;
		}
		CodeGenTrieNode newNode;
		for(int i = rci; i < keys.size(); i++) {
			newNode = new CodeGenTrieNode(i == keys.size() - 1, index, valueType, keys.get(i), new HashSet<>(), root.getType());
			newNode.setRowIndexBeginPos(properties.getRowIndexBegin());
			currentNode.getChildren().put(keys.get(i), newNode);
			currentNode = newNode;
		}
	}

	public String getJavaCode(){
		StringBuilder src = new StringBuilder();
		switch(properties.getRowIndex()){
			case IDENTIFY:
				getJavaCode(rootCol, src, "0");
				src.append("row++; \n");
				break;
			case PREFIX:
				getJavaCode(rootRow, src, "0");
				getJavaCode(rootCol, src, "0");
				break;
			case KEY:

				getJavaCode(rootCol, src, "0");
				break;
		}

		return src.toString();
	}


	public String getRandomName(String base) {
		Random r = new Random();
		int low = 0;
		int high = 100000000;
		int result = r.nextInt(high - low) + low;

		return base + "_" + result;
	}

	private void getJavaCode(CodeGenTrieNode node, StringBuilder src, String currPos){
		if(node.isEndOfCondition())
			src.append(node.geValueCode(destination, currPos));

		if(node.getChildren().size() > 0) {
			String currPosVariable;
			for(String key : node.getChildren().keySet()) {
				if(key.length() > 0) {
					currPosVariable = getRandomName("curPos");
					if(node.getKey() == null)
						src.append("index = str.indexOf(\"" + key.replace("\"", "\\\"") + "\"); \n");
					else
						src.append("index = str.indexOf(\"" + key.replace("\"", "\\\"") + "\", " + currPos + "); \n");
					src.append("if(index != -1) { \n");
					src.append("int " + currPosVariable + " = index + " + key.length() + "; \n");
					currPos = currPosVariable;
				}
				CodeGenTrieNode child = node.getChildren().get(key);
				getJavaCode(child, src, currPos);
				if(key.length() > 0)
					src.append("} \n");
			}
		}
	}
}
