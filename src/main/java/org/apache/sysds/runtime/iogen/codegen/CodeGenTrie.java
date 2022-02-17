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
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.iogen.CustomProperties;
import org.apache.sysds.runtime.iogen.KeyTrie;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

public class CodeGenTrie {
	private final CodeGenTrieNode rootCol;
	private final CodeGenTrieNode rootRow;
	private final CustomProperties properties;
	private final String destination;
	private HashMap<String, Integer> colKeyPatternMap;
	private HashSet<String> regexSet;
	private final boolean isRegexBase;
	private boolean isMatrix;

	public CodeGenTrie(CustomProperties properties, String destination) {
		this.rootCol = new CodeGenTrieNode(CodeGenTrieNode.NodeType.COL);
		this.rootRow = new CodeGenTrieNode(CodeGenTrieNode.NodeType.ROW);
		this.properties = properties;
		this.destination = destination;
		this.isMatrix = false;

		HashSet<String> conditions = new HashSet<>();
		for(int c = 0; c < properties.getColKeyPattern().length; c++) {
			KeyTrie keyTrie = properties.getColKeyPattern()[c];
			if(keyTrie != null) {
				for(ArrayList<String> keys : keyTrie.getReversePrefixKeyPatterns()) {
					conditions.add(keys.get(0));
					break;
				}
			}
		}
		if(conditions.size() < 2) {
			buildPrefixTree();
			this.isRegexBase = false;
		}
		else {
			this.colKeyPatternMap = new HashMap<>();
			this.regexSet = new HashSet<>();
			this.isRegexBase = true;
			buildPrefixTreeRegex();
		}

	}

	// Build Trie for Col and Row Key Patterns
	private void buildPrefixTreeRegex() {
		for(int c = 0; c < properties.getColKeyPattern().length; c++) {
			KeyTrie keyTrie = properties.getColKeyPattern()[c];
			if(keyTrie != null) {
				StringBuilder ksb = new StringBuilder();
				StringBuilder sbr = new StringBuilder();
				for(ArrayList<String> keys : keyTrie.getReversePrefixKeyPatterns()) {
					for(String ks : keys) {
						ksb.append(ks).append(Lop.OPERAND_DELIMITOR);
						String tmp = ks.replaceAll("\\s+", "\\\\s+");
						tmp = tmp.replaceAll("\\d+", "\\\\d+");
						sbr.append("(").append(tmp).append(")").append(Lop.OPERAND_DELIMITOR);
					}
					ksb.deleteCharAt(ksb.length() - 1);
					sbr.deleteCharAt(sbr.length() - 1);
					break;
				}
				if(ksb.length() == 0)
					colKeyPatternMap.put("", c);
				else
					colKeyPatternMap.put(ksb.toString(), c);
				regexSet.add(sbr.toString());

			}
		}

		if(properties.getRowIndex() == CustomProperties.IndexProperties.PREFIX) {
			KeyTrie keyTrie = properties.getRowKeyPattern();
			Types.ValueType vt = Types.ValueType.FP32;
			if(keyTrie != null) {
				for(ArrayList<String> keys : keyTrie.getReversePrefixKeyPatterns())
					this.insert(rootRow, -1, vt, keys);
			}
		}
	}

	// Build Trie for Col and Row Key Patterns
	private void buildPrefixTree() {
		for(int c = 0; c < properties.getColKeyPattern().length; c++) {
			KeyTrie keyTrie = properties.getColKeyPattern()[c];
			Types.ValueType vt = properties.getSchema() == null ? Types.ValueType.FP64 : properties.getSchema()[c];
			//Gson gson = new Gson();
			if(keyTrie != null) {
				//System.out.println(gson.toJson(keyTrie.getReversePrefixKeyPatterns()));
				for(ArrayList<String> keys : keyTrie.getReversePrefixKeyPatterns())
					this.insert(rootCol, c, vt, keys);
			}
			//else
			//	System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>> "+c);
		}

		if(properties.getRowIndex() == CustomProperties.IndexProperties.PREFIX) {
			KeyTrie keyTrie = properties.getRowKeyPattern();
			Types.ValueType vt = Types.ValueType.FP32;
			if(keyTrie != null) {
				for(ArrayList<String> keys : keyTrie.getReversePrefixKeyPatterns())
					this.insert(rootRow, -1, vt, keys);
			}
		}
	}

	private void insert(CodeGenTrieNode root, int index, Types.ValueType valueType, ArrayList<String> keys) {
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
		if(rci == keys.size()) {
			currentNode.setEndOfCondition(true);
			currentNode.setColIndex(index);
		}
		else {
			CodeGenTrieNode newNode;
			for(int i = rci; i < keys.size(); i++) {
				newNode = new CodeGenTrieNode(i == keys.size() - 1, index, valueType, keys.get(i), new HashSet<>(),
					root.getType());
				newNode.setRowIndexBeginPos(properties.getRowIndexBegin());
				currentNode.getChildren().put(keys.get(i), newNode);
				currentNode = newNode;
			}
		}
	}

	public String getJavaCode() {
		StringBuilder src = new StringBuilder();
		switch(properties.getRowIndex()) {
			case IDENTIFY:
				getJavaCode(rootCol, src, "0");
				src.append("row++; \n");
				break;
			case PREFIX:
				getJavaCode(rootRow, src, "0");
				getJavaCode(rootCol, src, "0");
				break;
			case KEY:
				src.append("String strChunk, remainedStr = null; \n");
				src.append("int chunkSize = 2048; \n");
				src.append("int recordIndex = 0; \n");
				src.append("try { \n");
				src.append("do{ \n");
				src.append("strChunk = getStringChunkOfBufferReader(br, remainedStr, chunkSize); \n");
				src.append("if(strChunk == null || strChunk.length() == 0) break; \n");
				src.append("do { \n");
				ArrayList<ArrayList<String>> kp = properties.getRowKeyPattern().getPrefixKeyPatterns();
				getJavaRowCode(src, kp, kp);
				getJavaCode(rootCol, src, "0");
				src.append("row++; \n");
				src.append("}while(true); \n");
				src.append("remainedStr = strChunk.substring(recordIndex); \n");

				src.append("}while(true); \n");
				src.append("} \n");
				src.append("finally { \n");
				src.append("IOUtilFunctions.closeSilently(br); \n");
				src.append("} \n");
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

	private void getJavaCode(CodeGenTrieNode node, StringBuilder src, String currPos) {
		if(!isRegexBase)
			getJavaCodeIndexOf(node, src, currPos);
		else
			getJavaCodeRegex(src);
	}

	private void getJavaCodeRegex(StringBuilder src) {

		// TODO: for fist item start with ""
		//src.append("List<String> allMatches = new ArrayList<String>(); \n");
		for(String s : regexSet) {
			if(s.equals("()")) {
				src.append("int colIndex0 = getColIndex(colKeyPatternMap, \"\"); \n");
				src.append("endPos = getEndPos(str, strLen, 0, endWithValueString[colIndex0]); \n");
				src.append("String cellStr0 = str.substring(0, endPos); \n");
				src.append("if ( cellStr0.length() > 0 ){\n");
				if(isMatrix) {
					src.append("Double cellValue0; \n");
					src.append("try{cellValue0 = Double.parseDouble(cellStr0); } catch(Exception e){cellValue0= 0d;}\n");
					src.append("if(cellValue0!= 0) { \n");
					src.append(destination).append("(row, colIndex0 , cellValue0); \n");
					src.append("lnnz++;\n");
					src.append("} \n");
				}
				else {
					src.append(destination).append("(row, colIndex0 , UtilFunctions.stringToObject(properties.getSchema()[colIndex0], cellStr)0); \n");
				}
				src.append("}\n");
			}
			else {
				int groupCount = s.split(Lop.OPERAND_DELIMITOR).length;
				if(groupCount > 1)
					break;
				src.append("Matcher matcher = Pattern.compile(\"" + s.replace("\\", "\\\\") + "\").matcher(str); \n");
				src.append("while(matcher.find()) { \n");
				src.append("String key = ").append("matcher.group(1);\n");
				src.append("int currPos = matcher.end();\n");
				src.append("int colIndex = getColIndex(colKeyPatternMap, key); \n");
				src.append("if(colIndex!=-1) { \n");
				//src.append("Types.ValueType vt = pair.getValue();\n");
				src.append("endPos = getEndPos(str, strLen, currPos, endWithValueString[colIndex]); \n");
				src.append("String cellStr = str.substring(currPos, endPos); \n");
				src.append("if ( cellStr.length() > 0 ){\n");
				if(isMatrix) {
					src.append("Double cellValue; \n");
					src.append("try{cellValue = Double.parseDouble(cellStr); } catch(Exception e){cellValue= 0d;}\n");
					src.append("if(cellValue!= 0) { \n");
					src.append(destination).append("(row, colIndex , cellValue); \n");
					src.append("lnnz++;\n");
					src.append("} \n");
				}
				else {
					src.append(destination).append("(row, colIndex , UtilFunctions.stringToObject(properties.getSchema()[colIndex], cellStr)); \n");
				}
				src.append("}\n");
				src.append("}\n");
				src.append("}\n");
			}
		}
	}

	private void getJavaCodeIndexOf(CodeGenTrieNode node, StringBuilder src, String currPos) {
		if(node.isEndOfCondition())
			src.append(node.geValueCode(destination, currPos));

		if(node.getChildren().size() > 0) {
			String currPosVariable = currPos;
			for(String key : node.getChildren().keySet()) {
				if(key.length() > 0) {
					currPosVariable = getRandomName("curPos");
					if(node.getKey() == null)
						src.append(
							"index = str.indexOf(\"" + key.replace("\\\"", "\"").replace("\"", "\\\"") + "\"); \n");
					else
						src.append("index = str.indexOf(\"" + key.replace("\\\"", "\"")
							.replace("\"", "\\\"") + "\", " + currPos + "); \n");
					src.append("if(index != -1) { \n");
					src.append("int " + currPosVariable + " = index + " + key.length() + "; \n");
				}
				CodeGenTrieNode child = node.getChildren().get(key);
				getJavaCodeIndexOf(child, src, currPosVariable);
				if(key.length() > 0)
					src.append("} \n");
			}
		}
	}

	private void getJavaRowCode(StringBuilder src, ArrayList<ArrayList<String>> rowBeginPattern,
		ArrayList<ArrayList<String>> rowEndPattern) {

		// TODO: we have to extend it to multi patterns
		// now, we assumed each row can have single pattern for begin and end

		for(ArrayList<String> kb : rowBeginPattern) {
			for(String k : kb) {
				src.append("recordIndex = strChunk.indexOf(\"" + k + "\", recordIndex); \n");
				src.append("if(recordIndex == -1) break; \n");
			}
			src.append("recordIndex +=" + kb.get(kb.size() - 1).length() + "; \n");
			break;
		}
		src.append("int recordBeginPos = recordIndex; \n");
		String endKey = rowEndPattern.get(0).get(0);
		src.append("recordIndex = strChunk.indexOf(\"" + endKey + "\", recordBeginPos);");
		src.append("if(recordIndex == -1) break; \n");
		src.append("str = strChunk.substring(recordBeginPos, recordIndex); \n");
		src.append("strLen = str.length(); \n");
	}

	public void setMatrix(boolean matrix) {
		isMatrix = matrix;
	}

	public boolean isRegexBase() {
		return isRegexBase;
	}

	public HashMap<String, Integer> getColKeyPatternMap() {
		return colKeyPatternMap;
	}
}
