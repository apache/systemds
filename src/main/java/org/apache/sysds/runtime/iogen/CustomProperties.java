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
import org.apache.sysds.runtime.io.FileFormatProperties;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;

public class CustomProperties extends FileFormatProperties implements Serializable {

	private MappingProperties mappingProperties;
	private RowIndexStructure rowIndexStructure;
	private ColIndexStructure colIndexStructure;

	public CustomProperties(MappingProperties mappingProperties, RowIndexStructure rowIndexStructure, ColIndexStructure colIndexStructure) {
		this.mappingProperties = mappingProperties;
		this.rowIndexStructure = rowIndexStructure;
		this.colIndexStructure = colIndexStructure;
	}

	public MappingProperties getMappingProperties() {
		return mappingProperties;
	}

	public void setMappingProperties(MappingProperties mappingProperties) {
		this.mappingProperties = mappingProperties;
	}

	public RowIndexStructure getRowIndexStructure() {
		return rowIndexStructure;
	}

	public void setRowIndexStructure(RowIndexStructure rowIndexStructure) {
		this.rowIndexStructure = rowIndexStructure;
	}

	public ColIndexStructure getColIndexStructure() {
		return colIndexStructure;
	}

	public void setColIndexStructure(ColIndexStructure colIndexStructure) {
		this.colIndexStructure = colIndexStructure;
	}



	//--------------------------------------

	public enum IndexProperties {
		IDENTITY, EXIST, SEQSCATTER, XARRAY, YARRAY,
		IDENTIFY, PREFIX, KEY;
		@Override
		public String toString() {
			return this.name().toUpperCase();
		}
	}





	private KeyTrie[] colKeyPattern;
	private Types.ValueType[] schema;
	private IndexProperties rowIndex;
	private KeyTrie rowKeyPattern;
	private String rowIndexBegin;
	private HashMap<String, Integer> colKeyPatternMap;

	public CustomProperties(KeyTrie[] colKeyPattern, IndexProperties rowIndex) {
		this.colKeyPattern = colKeyPattern;
		this.rowIndex = rowIndex;
	}

	public CustomProperties(KeyTrie[] colKeyPattern, KeyTrie rowKeyPattern) {
		this.colKeyPattern = colKeyPattern;
		this.rowIndex = IndexProperties.KEY;
		this.rowKeyPattern = rowKeyPattern;
	}

	public CustomProperties(KeyTrie[] colKeyPattern, IndexProperties rowIndex, KeyTrie rowKeyPattern) {
		this.colKeyPattern = colKeyPattern;
		this.rowIndex = rowIndex;
		this.rowKeyPattern = rowKeyPattern;
	}

	public KeyTrie[] getColKeyPattern() {
		return colKeyPattern;
	}

	public HashSet<String>[] endWithValueStrings() {
		HashSet<String>[] endWithValueString = new HashSet[colKeyPattern.length];
		for(int i = 0; i < colKeyPattern.length; i++)
			if(colKeyPattern[i] != null)
				endWithValueString[i] = colKeyPattern[i].getFirstSuffixKeyPatterns();
		return endWithValueString;
	}

	public HashSet<String> endWithValueStringsRow() {
		return rowKeyPattern.getFirstSuffixKeyPatterns();
	}

	public void setColKeyPattern(KeyTrie[] colKeyPattern) {
		this.colKeyPattern = colKeyPattern;
	}

	public Types.ValueType[] getSchema() {
		return schema;
	}

	public void setSchema(Types.ValueType[] schema) {
		this.schema = schema;
	}

	public IndexProperties getRowIndex() {
		return rowIndex;
	}

	public void setRowIndex(IndexProperties rowIndex) {
		this.rowIndex = rowIndex;
	}

	public KeyTrie getRowKeyPattern() {
		return rowKeyPattern;
	}

	public void setRowKeyPattern(KeyTrie rowKeyPattern) {
		this.rowKeyPattern = rowKeyPattern;
	}

	public String getRowIndexBegin() {
		return rowIndexBegin;
	}

	public void setRowIndexBegin(String rowIndexBegin) {
		this.rowIndexBegin = rowIndexBegin;
	}

	public HashMap<String, Integer> getColKeyPatternMap() {
		return colKeyPatternMap;
	}

	public void setColKeyPatternMap(HashMap<String, Integer> colKeyPatternMap) {
		this.colKeyPatternMap = colKeyPatternMap;
	}
}
