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
import org.apache.sysds.runtime.iogen.FormatIdentifyer;
import org.apache.sysds.runtime.iogen.MappingProperties;
import org.apache.sysds.runtime.iogen.RowIndexStructure;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

public class CodeGenFlat {
	private final CustomProperties properties;
	private final CodeGenTrieNode[] ctnValue;
	private final CodeGenTrieNode ctnIndexes;
	private final String destination;
	private boolean isMatrix;

	private final FormatIdentifyer formatIdentifyer;

	public CodeGenFlat(CustomProperties properties, String destination, boolean isMatrix,
		FormatIdentifyer formatIdentifyer) {
		this.properties = properties;
		this.destination = destination;
		this.isMatrix = isMatrix;
		this.formatIdentifyer = formatIdentifyer;

		this.ctnValue = new CodeGenTrieNode[properties.getNcols()];
		this.ctnIndexes = new CodeGenTrieNode(CodeGenTrieNode.NodeType.INDEX);

		if(properties.getColKeyPatterns() != null) {
			for(int c = 0; c < properties.getColKeyPatterns().length; c++) {
				Types.ValueType vt = Types.ValueType.FP64;
				if(!this.isMatrix)
					vt = properties.getSchema()[c];
				this.ctnValue[c] = new CodeGenTrieNode(CodeGenTrieNode.NodeType.VALUE);
				this.insert(ctnValue[c], c + "", vt, properties.getColKeyPatterns()[c]);
			}
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
				newNode = new CodeGenTrieNode(i == keys.size() - 1, index, valueType, keys.get(i), new HashSet<>(),
					root.getType());
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
			for(int c=0; c< properties.getNcols(); c++)
				getJavaCode(ctnValue[c], src, "0", false);
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

	private void getJavaCode(CodeGenTrieNode node, StringBuilder src, String currPos, boolean arrayCodeGenEnable) {
		getJavaCodeIndexOf(node, src, currPos, arrayCodeGenEnable);
	}

	private void getJavaCodeIndexOf(CodeGenTrieNode node, StringBuilder src, String currPos,
		boolean arrayCodeGenEnable) {
		CodeGenTrieNode tmpNode = null;
		if(arrayCodeGenEnable)
			tmpNode = getJavaCodeRegular(node, src, currPos);
		if(tmpNode == null) {
			if(node.isEndOfCondition())
				src.append(node.geValueCode(destination, currPos));

			if(node.getChildren().size() > 0) {
				String currPosVariable = currPos;
				for(String key : node.getChildren().keySet()) {
					if(key.length() > 0) {
						currPosVariable = getRandomName("curPos");
						String mKey = key.replace("\\\"", Lop.OPERAND_DELIMITOR);
						mKey = mKey.replace("\\", "\\\\");
						mKey = mKey.replace(Lop.OPERAND_DELIMITOR, "\\\"");
						if(node.getKey() == null) {
							src.append("index = str.indexOf(\"" + mKey.replace("\\\"", "\"").replace("\"", "\\\"") +
								"\"); \n");
						}
						else
							src.append(
								"index = str.indexOf(\"" + mKey.replace("\\\"", "\"").replace("\"", "\\\"") + "\", " +
									currPos + "); \n");
						src.append("if(index != -1) { \n");
						src.append("int " + currPosVariable + " = index + " + key.length() + "; \n");
					}
					CodeGenTrieNode child = node.getChildren().get(key);
					getJavaCodeIndexOf(child, src, currPosVariable, arrayCodeGenEnable);
					if(key.length() > 0)
						src.append("} \n");
				}
			}
		}
		else if(!tmpNode.isEndOfCondition())
			getJavaCodeIndexOf(tmpNode, src, currPos, arrayCodeGenEnable);
	}

	private CodeGenTrieNode getJavaCodeRegular(CodeGenTrieNode node, StringBuilder src, String currPos) {
		ArrayList<CodeGenTrieNode> nodes = new ArrayList<>();
		if(node.getChildren().size() == 1) {
			nodes.add(node);
			CodeGenTrieNode cn = node.getChildren().get(node.getChildren().keySet().iterator().next());
			do {
				if(cn.getChildren().size() <= 1) {
					nodes.add(cn);
					if(cn.getChildren().size() == 1)
						cn = cn.getChildren().get(cn.getChildren().keySet().iterator().next());
					else
						break;
				}
				else
					break;
			}
			while(true);
			if(nodes.size() > 1) {
				boolean isKeySingle;
				boolean isIndexSequence = true;

				// extract keys and related indexes
				ArrayList<String> keys = new ArrayList<>();
				ArrayList<String> colIndexes = new ArrayList<>();
				ArrayList<Integer> colIndexesExtra = new ArrayList<>();
				int tmpIndex = 0;
				for(CodeGenTrieNode n : nodes) {
					keys.add(n.getKey());
					if(n.isEndOfCondition())
						colIndexes.add(n.getColIndex());
					else
						colIndexesExtra.add(tmpIndex);
					tmpIndex++;
				}
				if(keys.size() != colIndexes.size()) {
					if(keys.size() == colIndexes.size() + 1 && colIndexesExtra.get(0) == 0) {}
					else
						return null;
				}
				// are keys single?
				HashSet<String> keysSet = new HashSet<>();
				for(int i = 1; i < keys.size(); i++)
					keysSet.add(keys.get(i));
				isKeySingle = keysSet.size() == 1;

				for(int i = 1; i < colIndexes.size() && isIndexSequence; i++) {
					isIndexSequence =
						Integer.parseInt(colIndexes.get(i)) - Integer.parseInt(colIndexes.get(i - 1)) == 1;
				}
				// Case 1: key = single and index = sequence
				// Case 2: key = single and index = irregular
				// Case 3: key = multi and index = sequence
				// Case 4: key = multi and index = irregular

				String tmpDest = destination.split("\\.")[0];

				int[] cols = new int[colIndexes.size()];
				for(int i = 0; i < cols.length; i++)
					cols[i] = Integer.parseInt(colIndexes.get(i));

				// check 1:
				String conflict = !isMatrix ? formatIdentifyer.getConflictToken(cols) : null;

				// check is array has conflict?
				// if the array has just one item, the if-then-else is a good option
				// otherwise we will follow loop code gen
				if(colIndexes.size() == 1) {
					if(conflict != null) {
						src.append("// conflict token : " + conflict + " appended to end of value token list \n");
						properties.endWithValueStrings()[Integer.parseInt(colIndexes.get(0))].add(conflict);
					}
					else
						src.append("// conflict token for find end of array was NULL \n");
					//getJavaCodeIndexOf(node, src, currPos, false);
				}
				else {
					boolean isDelimAndSuffixesSame = false;
					// #Case 1: key = single and index = sequence
					if(isKeySingle && isIndexSequence) {
						String baseIndex = colIndexes.get(0);
						String key = keysSet.iterator().next();
						String mKey = refineKeyForSearch(key);
						String colIndex = getRandomName("colIndex");
						if(!isMatrix) {
							isDelimAndSuffixesSame = formatIdentifyer.isDelimAndSuffixesSame(key, cols, conflict);
							if(conflict != null) {
								src.append("indexConflict=")
									.append("str.indexOf(" + refineKeyForSearch(conflict) + "," + currPos + "); \n");
								src.append("if (indexConflict != -1) \n");
								src.append(
									"parts = IOUtilFunctions.splitCSV(str.substring(" + currPos + ", indexConflict)," +
										mKey + "); \n");
								src.append("else \n");
							}
							src.append(
								"parts=IOUtilFunctions.splitCSV(str.substring(" + currPos + "), " + mKey + "); \n");
							src.append("int ").append(colIndex).append("; \n");
							src.append("for (int i=0; i< Math.min(parts.length, " + colIndexes.size() + "); i++) {\n");
							src.append(colIndex).append(" = i+").append(baseIndex).append("; \n");
							if(isDelimAndSuffixesSame) {
								if(!isMatrix)
									src.append(destination).append(
										"(row," + colIndex + ",UtilFunctions.stringToObject(" + tmpDest +
											".getSchema()[" + colIndex + "], parts[i])); \n");
								else
									src.append(destination).append(
										"(row," + colIndex + ",UtilFunctions.parseToDouble(parts[i], null)); \n");
							}
							else {
								src.append(
									"endPos=TemplateUtil.getEndPos(parts[i], parts[i].length(),0,endWithValueString[" +
										colIndex + "]); \n");
								if(!isMatrix)
									src.append(destination).append(
										"(row," + colIndex + ",UtilFunctions.stringToObject(" + tmpDest +
											".getSchema()[" + colIndex + "], parts[i].substring(0,endPos))); \n");
								else
									src.append(destination).append("(row," + colIndex +
										",UtilFunctions.parseToDouble(parts[i].substring(0,endPos), null)); \n");
							}
							src.append("} \n");
							if(conflict != null) {
								src.append("if (indexConflict !=-1) \n");
								src.append("index = indexConflict; \n");
							}
						}

						return cn;
					}
					// #Case 2: key = single and index = irregular
					if(isKeySingle && !isIndexSequence) {
						StringBuilder srcColIndexes = new StringBuilder("new int[]{");
						for(String c : colIndexes)
							srcColIndexes.append(c).append(",");

						srcColIndexes.deleteCharAt(srcColIndexes.length() - 1);
						srcColIndexes.append("}");
						String colIndexName = getRandomName("targetColIndex");
						src.append("int[] ").append(colIndexName).append("=").append(srcColIndexes).append("; \n");
						String key = keysSet.iterator().next();
						String mKey = refineKeyForSearch(key);

						if(!isMatrix) {
							isDelimAndSuffixesSame = formatIdentifyer.isDelimAndSuffixesSame(key, cols, conflict);
							if(conflict != null) {
								src.append("indexConflict = ")
									.append("str.indexOf(" + refineKeyForSearch(conflict) + "," + currPos + "); \n");
								src.append("if (indexConflict != -1) \n");
								src.append(
									"parts = IOUtilFunctions.splitCSV(str.substring(" + currPos + ", indexConflict), " +
										mKey + "); \n");
								src.append("else \n");
							}
						}
						src.append(
							"parts = IOUtilFunctions.splitCSV(str.substring(" + currPos + "), " + mKey + "); \n");
						src.append("for (int i=0; i< Math.min(parts.length, " + colIndexes.size() + "); i++) {\n");
						if(isDelimAndSuffixesSame) {
							if(!isMatrix) {
								src.append(destination).append(
									"(row," + colIndexName + "[i],UtilFunctions.stringToObject(" + tmpDest +
										".getSchema()[" + colIndexName + "[i]], parts[i])); \n");
							}
							else
								src.append(destination).append(
									"(row," + colIndexName + "[i],UtilFunctions.parseToDouble(parts[i], null)); \n");
						}
						else {
							if(!isMatrix) {
								src.append(
									"endPos = TemplateUtil.getEndPos(parts[i], parts[i].length(), 0, endWithValueString[" +
										colIndexName + "[i]]); \n");
								src.append(destination).append(
									"(row," + colIndexName + "[i],UtilFunctions.stringToObject(" + tmpDest +
										".getSchema()[" + colIndexName + "[i]], parts[i].substring(0, endPos))); \n");
							}
							else
								src.append(destination).append("(row," + colIndexName +
									"[i],UtilFunctions.parseToDouble(parts[i].substring(0, endPos), null)); \n");
						}
						src.append("} \n");
						if(conflict != null) {
							src.append("if (indexConflict !=-1) \n");
							src.append("index = indexConflict; \n");
						}
						return cn;
					}
					// #Case 3: key = multi and index = sequence
					// #Case 4: key = multi and index = irregular
					else
						return null;
				}
			}
			else
				return null;
		}
		return null;
	}

	private String refineKeyForSearch(String k) {
		String mKey = k.replace("\\\"", Lop.OPERAND_DELIMITOR);
		mKey = mKey.replace("\\", "\\\\");
		mKey = mKey.replace(Lop.OPERAND_DELIMITOR, "\\\"");
		mKey = "\"" + mKey.replace("\\\"", "\"").replace("\"", "\\\"") + "\"";
		return mKey;
	}

	public void setMatrix(boolean matrix) {
		isMatrix = matrix;
	}
}
