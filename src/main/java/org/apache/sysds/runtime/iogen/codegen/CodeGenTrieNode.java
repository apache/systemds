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

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

public class CodeGenTrieNode {

	public enum NodeType {
		VALUE, ROW, COL, INDEX;

		@Override public String toString() {
			return this.name().toUpperCase();
		}
	}

	private final Map<String, CodeGenTrieNode> children = new HashMap<>();
	private boolean endOfCondition;
	private String colIndex;
	private Types.ValueType valueType;
	private String key;
	private HashSet<String> naStrings;
	private final NodeType type;
	private int rowIndexBeginPos;
	private int colIndexBeginPos;

	public CodeGenTrieNode(NodeType type) {
		this.endOfCondition = false;
		this.type = type;
	}

	public CodeGenTrieNode(String colIndex, String key, NodeType type) {
		this.colIndex = colIndex;
		this.key = key;
		this.type = type;
	}

	public CodeGenTrieNode(boolean endOfCondition, String colIndex, Types.ValueType valueType, String key, HashSet<String> naStrings, NodeType type) {
		this.endOfCondition = endOfCondition;
		this.colIndex = colIndex;
		this.valueType = valueType;
		this.key = key;
		if(endOfCondition) {
			this.naStrings = naStrings;
		}
		this.type = type;
	}

	public String geValueCode(String destination, String currPos) {
		if(this.type == NodeType.INDEX)
			return this.getIndexCode(currPos);
		else
			return this.getColValueCode(destination, currPos);
	}

	private String getIndexCode(String currPos) {
		StringBuilder src = new StringBuilder();
		String subStr;
		String ewvs;
		if(this.colIndex.equals("0"))
			ewvs = "endWithValueStringRow";
		else
			ewvs = "endWithValueStringCol";

		src.append("endPos = TemplateUtil.getEndPos(str, strLen, " + currPos + "," + ewvs + "); \n");
		subStr = "str.substring(" + currPos + ",endPos)";
		src.append("try{ \n");
		if(this.colIndex.equals("0")) {
			if(rowIndexBeginPos > 0)
				src.append("row = ").append("Integer.parseInt(" + subStr + ") - " + rowIndexBeginPos + "; \n");
			else
				src.append("row = ").append("Integer.parseInt(" + subStr + "); \n");
		}
		else {
			if(colIndexBeginPos > 0)
				src.append("col = ").append("Integer.parseInt(" + subStr + ") - " + colIndexBeginPos + "; \n");
			else
				src.append("col = ").append("Integer.parseInt(" + subStr + "); \n");
		}
		src.append("} catch(Exception e){} \n");
		return src.toString();
	}


	private String getColValueCode(String destination, String currPos) {

		StringBuilder src = new StringBuilder();
		if(this.colIndex.equals("col"))
			src.append("endPos = TemplateUtil.getEndPos(str, strLen, " + currPos + ", endWithValueStringVal); \n");
		else
			src.append("endPos = TemplateUtil.getEndPos(str, strLen, " + currPos + ", endWithValueString[" + colIndex + "]); \n");

		src.append("String cellStr" + colIndex + " = str.substring(" + currPos + ",endPos); \n");

		if(valueType.isNumeric()) {
			src.append("if ( cellStr" + colIndex + ".length() > 0 ){\n");
			src.append(getParsCode("cellStr" + colIndex));
			src.append("if(cellValue" + colIndex + " != 0) { \n");
			src.append(destination).append("(row, " + colIndex + ", cellValue" + colIndex + "); \n");
			src.append("lnnz++;\n");
			src.append("}\n");
			src.append("}\n");
		}
		else if(valueType == Types.ValueType.STRING || valueType == Types.ValueType.BOOLEAN) {
			if(naStrings != null && naStrings.size() > 0) {
				StringBuilder sb = new StringBuilder();
				sb.append("if(");
				for(String na : naStrings) {
					src.append("naStrings.contains(\"" + na + "\")").append("|| \n");
				}
				sb.delete(sb.length() - 2, sb.length());
				sb.append("){ \n");
				sb.append("cellValue+" + colIndex + " = null;");
				sb.append("}\n");
			}
			else
				src.append(getParsCode("cellStr" + colIndex));
			src.append(destination).append("(row, " + colIndex + ", cellValue" + colIndex + "); \n");
		}
		return src.toString();
	}

	private String getParsCode(String subStr) {
		String cellValue = "cellValue" + colIndex;
		switch(valueType) {
			case STRING:
				return "String " + cellValue + " = " + subStr + "; \n";
			case BOOLEAN:
				return "Boolean " + cellValue + "; \n try{ " + cellValue + "= Boolean.parseBoolean(" + subStr + ");} catch(Exception e){" + cellValue + "=false;} \n";
			case INT32:
				return "Integer " + cellValue + "; \n try{ " + cellValue + "= Integer.parseInt(" + subStr + ");} catch(Exception e){" + cellValue + " = 0;} \n";
			case INT64:
				return "Long " + cellValue + "; \n try{" + cellValue + "= Long.parseLong(" + subStr + "); } catch(Exception e){" + cellValue + " = 0l;} \n";
			case FP64:
				return "Double " + cellValue + "; \n try{ " + cellValue + "= Double.parseDouble(" + subStr + "); } catch(Exception e){" + cellValue + " = 0d;}\n";
			case FP32:
				return "Float " + cellValue + "; \n try{ " + cellValue + "= Float.parseFloat(" + subStr + ");} catch(Exception e){" + cellValue + " = 0f;} \n";
			default:
				throw new RuntimeException("Unsupported value type: " + valueType);
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

	public String getColIndex() {
		return colIndex;
	}

	public void setColIndex(String colIndex) {
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

	public NodeType getType() {
		return type;
	}

	public int getRowIndexBeginPos() {
		return rowIndexBeginPos;
	}

	public void setRowIndexBeginPos(int rowIndexBeginPos) {
		this.rowIndexBeginPos = rowIndexBeginPos;
	}

	public int getColIndexBeginPos() {
		return colIndexBeginPos;
	}

	public void setColIndexBeginPos(int colIndexBeginPos) {
		this.colIndexBeginPos = colIndexBeginPos;
	}
}
