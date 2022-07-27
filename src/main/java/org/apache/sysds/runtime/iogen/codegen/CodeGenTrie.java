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
import org.apache.sysds.runtime.iogen.ColIndexStructure;
import org.apache.sysds.runtime.iogen.CustomProperties;
import org.apache.sysds.runtime.iogen.KeyTrie;
import org.apache.sysds.runtime.iogen.MappingProperties;
import org.apache.sysds.runtime.iogen.RowIndexStructure;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

public class CodeGenTrie {
	private final CustomProperties properties;
	private final CodeGenTrieNode ctnValue;
	private final CodeGenTrieNode ctnIndexes;

	private final String destination;
	private boolean isMatrix;

	public CodeGenTrie(CustomProperties properties, String destination, boolean isMatrix) {
		this.properties = properties;
		this.destination = destination;
		this.isMatrix = isMatrix;

		this.ctnValue = new CodeGenTrieNode(CodeGenTrieNode.NodeType.VALUE);
		this.ctnIndexes = new CodeGenTrieNode(CodeGenTrieNode.NodeType.INDEX);

		if(properties.getColKeyPatterns() != null) {
			for(int c = 0; c < properties.getColKeyPatterns().length; c++) {
				KeyTrie keyTrie = properties.getColKeyPatterns()[c];
				Types.ValueType vt = Types.ValueType.FP64;
				if(!this.isMatrix)
					vt = properties.getSchema()[c];
				if(keyTrie != null) {
					for(ArrayList<String> keys : keyTrie.getReversePrefixKeyPatterns())
						this.insert(ctnValue, c + "", vt, keys);
				}
			}
		}
		else if(properties.getValueKeyPattern() != null) {
			// TODO: same pattern for all columns but the ValueTypes are different- fix it !
			for(ArrayList<String> keys : properties.getValueKeyPattern().getPrefixKeyPatterns()) {
				this.insert(ctnValue, "col", Types.ValueType.FP64, keys);
			}
		}

		if(properties.getRowIndexStructure().getProperties() == RowIndexStructure.IndexProperties.RowWiseExist ||
			properties.getRowIndexStructure().getProperties() == RowIndexStructure.IndexProperties.CellWiseExist) {
			for(ArrayList<String> keys : properties.getRowIndexStructure().getKeyPattern().getPrefixKeyPatterns())
				this.insert(ctnIndexes, "0", Types.ValueType.INT32, keys);
		}

		if(properties.getColIndexStructure().getProperties() == ColIndexStructure.IndexProperties.CellWiseExist &&
			properties.getColIndexStructure().getKeyPattern() !=null) {
			for(ArrayList<String> keys : properties.getColIndexStructure().getKeyPattern().getPrefixKeyPatterns())
				this.insert(ctnIndexes, "1", Types.ValueType.INT32, keys);
		}
	}

	private void insert(CodeGenTrieNode root, String index, Types.ValueType valueType, ArrayList<String> keys) {
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
				newNode = new CodeGenTrieNode(i == keys.size() - 1, index, valueType, keys.get(i), new HashSet<>(), root.getType());
				newNode.setRowIndexBeginPos(properties.getRowIndexStructure().getRowIndexBegin());
				newNode.setColIndexBeginPos(properties.getColIndexStructure().getColIndexBegin());
				currentNode.getChildren().put(keys.get(i), newNode);
				currentNode = newNode;
			}
		}
	}

	public String getJavaCode() {
		StringBuilder src = new StringBuilder();
		int ncols = properties.getNcols();
		MappingProperties.DataProperties data = properties.getMappingProperties().getDataProperties();
		RowIndexStructure.IndexProperties rowIndex = properties.getRowIndexStructure().getProperties();
		ColIndexStructure.IndexProperties colIndex = properties.getColIndexStructure().getProperties();

		// example: csv
		if(data != MappingProperties.DataProperties.NOTEXIST &&
			((rowIndex == RowIndexStructure.IndexProperties.Identity &&
			colIndex == ColIndexStructure.IndexProperties.Identity) ||
			rowIndex == RowIndexStructure.IndexProperties.SeqScatter)) {
			getJavaCode(ctnValue, src, "0");
			src.append("row++; \n");
		}
		// example: MM
		else if(rowIndex == RowIndexStructure.IndexProperties.CellWiseExist &&
			colIndex == ColIndexStructure.IndexProperties.CellWiseExist) {
			getJavaCode(ctnIndexes, src, "0");
			src.append("if(col < " + ncols + "){ \n");
			if(data != MappingProperties.DataProperties.NOTEXIST) {
				getJavaCode(ctnValue, src, "0");
			}
			else
				src.append(destination).append("(row, col, cellValue); \n");
			src.append("} \n");
		}
		// example: LibSVM
		else if(rowIndex == RowIndexStructure.IndexProperties.Identity &&
			colIndex == ColIndexStructure.IndexProperties.CellWiseExist){
			src.append("String strValues[] = str.split(\""+ properties.getColIndexStructure().getValueDelim()+"\"); \n");
			src.append("for(String si: strValues){ \n");
			src.append("String strIndexValue[] = si.split(\""+ properties.getColIndexStructure().getIndexDelim()+"\", -1); \n");
			src.append("if(strIndexValue.length == 2){ \n");
			src.append("col = UtilFunctions.parseToInt(strIndexValue[0]); \n");
			src.append("if(col <= "+ncols+"){ \n");
			if(this.isMatrix){
				src.append("try{ \n");
				src.append(destination).append("(row, col, Double.parseDouble(strIndexValue[1])); \n");
				src.append("lnnz++;\n");
				src.append("} catch(Exception e){"+destination+"(row, col, 0d);} \n");
			}
			else {
				src.append(destination).append("(row, col, UtilFunctions.stringToObject(_props.getSchema()[col], strIndexValue[1]); \n");
			}
			src.append("} \n");
			src.append("} \n");
			src.append("} \n");
			src.append("row++; \n");
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
		getJavaCodeIndexOf(node, src, currPos);
	}

	private void getJavaCodeIndexOf(CodeGenTrieNode node, StringBuilder src, String currPos) {
		if(node.isEndOfCondition())
			src.append(node.geValueCode(destination, currPos));

		if(node.getChildren().size() > 0) {
			String currPosVariable = currPos;
			for(String key : node.getChildren().keySet()) {
				if(key.length() > 0) {
					currPosVariable = getRandomName("curPos");
					String mKey = key.replace("\\\"", Lop.OPERAND_DELIMITOR);
					mKey = mKey.replace("\\", "\\\\");
					mKey = mKey.replace(Lop.OPERAND_DELIMITOR,"\\\"");
					if(node.getKey() == null) {
						src.append("index = str.indexOf(\"" + mKey.replace("\\\"", "\"").replace("\"", "\\\"") + "\"); \n");
					}
					else
						src.append("index = str.indexOf(\"" + mKey.replace("\\\"", "\"").replace("\"", "\\\"") + "\", " + currPos + "); \n");
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

	public void setMatrix(boolean matrix) {
		isMatrix = matrix;
	}
}
