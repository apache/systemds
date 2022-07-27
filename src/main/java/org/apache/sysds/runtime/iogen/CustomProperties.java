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
import java.util.HashSet;

public class CustomProperties extends FileFormatProperties implements Serializable {

	private MappingProperties mappingProperties;
	private RowIndexStructure rowIndexStructure;
	private ColIndexStructure colIndexStructure;
	private KeyTrie[] colKeyPatterns;
	private KeyTrie valueKeyPattern;
	private Types.ValueType[] schema;
	private int ncols;
	private boolean sparse;
	private boolean parallel;

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

	public KeyTrie[] getColKeyPatterns() {
		return colKeyPatterns;
	}

	public void setColKeyPatterns(KeyTrie[] colKeyPatterns) {
		this.colKeyPatterns = colKeyPatterns;
	}

	public Types.ValueType[] getSchema() {
		return schema;
	}

	public void setSchema(Types.ValueType[] schema) {
		this.schema = schema;
	}

	public HashSet<String>[] endWithValueStrings() {
		if(colKeyPatterns !=null) {
			HashSet<String>[] endWithValueString = new HashSet[colKeyPatterns.length];
			for(int i = 0; i < colKeyPatterns.length; i++)
				if(colKeyPatterns[i] != null)
					endWithValueString[i] = colKeyPatterns[i].getFirstSuffixKeyPatterns();
			return endWithValueString;
		}
		else
			return null;
	}

	public KeyTrie getValueKeyPattern() {
		return valueKeyPattern;
	}

	public void setValueKeyPattern(KeyTrie valueKeyPattern) {
		this.valueKeyPattern = valueKeyPattern;
	}

	public int getNcols() {
		return ncols;
	}

	public void setNcols(int ncols) {
		this.ncols = ncols;
	}

	public boolean isSparse() {
		return sparse;
	}

	public void setSparse(boolean sparse) {
		this.sparse = sparse;
	}

	public boolean isParallel() {
		return parallel;
	}

	public void setParallel(boolean parallel) {
		this.parallel = parallel;
	}
}
