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

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class CodeGenTrieNode {

	private final Map<String, CodeGenTrieNode> children = new HashMap<>();
	private boolean endOfCondition;
	private int colIndex;
	private Types.ValueType valueType;
	private HashSet<String> endWithValueString;
	private String key;
	private HashSet<String> naStrings;

	public CodeGenTrieNode() {
		this.endOfCondition = false;
	}

	public CodeGenTrieNode(int colIndex, String key) {
		this.colIndex = colIndex;
		this.key = key;
	}

	public CodeGenTrieNode(boolean endOfCondition, int colIndex, Types.ValueType valueType, String key, HashSet<String> endWithValueString, HashSet<String> naStrings) {
		this.endOfCondition = endOfCondition;
		this.colIndex = colIndex;
		this.valueType = valueType;
		this.key = key;
		if(endOfCondition){
			this.endWithValueString = endWithValueString;
			this.naStrings = naStrings;
		}

	}

	public String geValueCode(String destination, String currPos){

		StringBuilder src = new StringBuilder();
		String subStr;

		if(this.endWithValueString.size() == 1) {
			String delim = this.endWithValueString.iterator().next();
			if(delim.length() > 0)
				subStr = "str.substring("+currPos+", str.indexOf(\""+delim+"\", "+currPos+"))";
			else
				subStr = "str.substring("+currPos+")";
		}
		else {
			int i = 0;
			for(String d: this.endWithValueString){
				if(i == 0) {
					if(d.length() == 0)
						src.append("endPos = strLen; \n");
					else
						src.append("endPos = str.indexOf(\"" + d + "\", "+currPos+"); \n");
				}
				else {
					if(d.length() == 0)
						src.append("endPos = Math.min(strLen, endPos); \n");
					else
						src.append("endPos = Math.min(endPos, str.indexOf(\"" + d + "\", "+currPos+")); \n");
				}
				i++;
			}
			subStr = "str.substring(currPos, endPos)";
		}
		if(valueType.isNumeric()) {
			src.append(getParsCode(subStr));
			src.append("if(cellValue"+colIndex+" != 0) { \n");
			src.append(destination).append("(row, " + colIndex + ", cellValue"+colIndex+"); \n");
			src.append("lnnz++;\n");
			src.append("}\n");
		}
		else if(valueType == Types.ValueType.STRING || valueType == Types.ValueType.BOOLEAN){
			if(naStrings.size() > 0) {
				StringBuilder sb = new StringBuilder();
				sb.append("if(");
				for(String na : naStrings) {
					src.append("naStrings.contains(\"" + na + "\")").append("|| \n");
				}
				sb.delete(sb.length()-2, sb.length());
				sb.append("){ \n");
				sb.append("cellValue+"+colIndex+" = null;");
				sb.append("}\n");
			}
			else
				src.append(getParsCode(subStr));
			src.append(destination).append("(row, " + colIndex + ", cellValue+"+colIndex+"); \n");
		}
		return src.toString();
	}

	private String getParsCode(String subStr) {
		switch(valueType ) {
			case STRING:  return "String cellValue"+colIndex+" = "+subStr+"; \n";
			case BOOLEAN: return "Boolean cellValue"+colIndex+" = Boolean.parseBoolean("+subStr+"); \n";
			case INT32:   return "Integer cellValue"+colIndex+" = Integer.parseInt("+subStr+"); \n";
			case INT64:   return "Long cellValue"+colIndex+" = Long.parseLong("+subStr+"); \n";
			case FP64:    return "Float cellValue"+colIndex+" = Double.parseDouble("+subStr+"); \n";
			case FP32:    return "Double cellValue"+colIndex+" = Float.parseFloat("+subStr+"); \n";
			default: throw new RuntimeException("Unsupported value type: "+valueType);
		}
	}


	public Map<String, CodeGenTrieNode> getChildren() {
		return children;
	}

	public boolean isEndOfCondition() {
		return endOfCondition;
	}

	public void setEndOfCondition(boolean endOfCondition) {
		this.endOfCondition = endOfCondition;
	}

	public int getColIndex() {
		return colIndex;
	}

	public void setColIndex(int colIndex) {
		this.colIndex = colIndex;
	}

	public Types.ValueType getValueType() {
		return valueType;
	}

	public void setValueType(Types.ValueType valueType) {
		this.valueType = valueType;
	}

	public String getKey() {
		return key;
	}

	public void setKey(String key) {
		this.key = key;
	}
}
