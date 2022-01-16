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
import org.apache.sysds.runtime.iogen.MappingTrieNode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;

public class CodeGenTrie {
	private CodeGenTrieNode root;

	public CodeGenTrie() {
		root = new CodeGenTrieNode();
	}

	public void insert(int colIndex, Types.ValueType valueType, ArrayList<String> keys, HashSet<String> endWithValueString) {
	
		CodeGenTrieNode currentNode = root;
		int index = 0;
		for(String key : keys) {
			if(currentNode.getChildren().containsKey(key)) {
				currentNode = currentNode.getChildren().get(key);
				index++;
			}
			else
				break;
		}

		CodeGenTrieNode newNode;
		for(int i = index; i < keys.size(); i++) {
			newNode = new CodeGenTrieNode(i == keys.size() - 1, colIndex, valueType, keys.get(i), endWithValueString, new HashSet<>());
			currentNode.getChildren().put(keys.get(i), newNode);
			currentNode = newNode;
		}
	}
	public String getJavaCode(){
		StringBuilder src = new StringBuilder();
		getJavaCode(root, src, "dest.appendValue", "0");
		return src.toString();
	}

	private String getRandomName(String base) {
		Random r = new Random();
		int low = 0;
		int high = 100000000;
		int result = r.nextInt(high - low) + low;

		return base + "_" + result;
	}
	private void getJavaCode(CodeGenTrieNode node, StringBuilder src, String destination, String currPos){
		String currPosVariable = getRandomName("curPos");
		if(node.getChildren().size() ==0){
			String key = node.getKey();
			if(key.length() > 0){
				src.append("index = str.indexOf(\""+node.getKey()+"\", "+currPos+"); \n");
				src.append("if(index != -1) { \n");
				src.append("int "+currPosVariable + " = index + "+ key.length()+"; \n");
				src.append(node.geValueCode(destination, currPosVariable));
				src.append("}\n");
			}
			else {
				src.append(node.geValueCode(destination, "0"));
			}
		}
		else {
			if(node.getKey()!=null) {
				src.append("index = str.indexOf(\"" + node.getKey() + "\", "+currPos+"); \n");
				src.append("if(index != -1) { \n");
				src.append("int "+currPosVariable + " = index + "+ node.getKey().length()+"; \n");
				currPos = currPosVariable;
			}

			for(String key: node.getChildren().keySet()){
				CodeGenTrieNode child = node.getChildren().get(key);
				getJavaCode(child, src, destination, currPos);
			}
			if(node.getKey()!=null){
				src.append("}\n");
			}
		}
	}
}
