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
import java.util.ArrayList;
import java.util.HashSet;

public class CustomProperties extends FileFormatProperties implements Serializable {
	private static final long serialVersionUID = -2877155313260718134L;
	
	private MappingProperties mappingProperties;
	private RowIndexStructure rowIndexStructure;
	private ColIndexStructure colIndexStructure;
	private HashSet<String>[] endWithValueStrings;
	private ArrayList<String>[] colKeyPatterns;
	private ArrayList<String> valueKeyPattern;
	private Types.ValueType[] schema;

	private int ncols;
	private boolean sparse;
	private boolean parallel;

	public CustomProperties(MappingProperties mappingProperties, RowIndexStructure rowIndexStructure,
		ColIndexStructure colIndexStructure) {
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

	public ArrayList<String>[] getColKeyPatterns() {
		return colKeyPatterns;
	}

	public void setColKeyPatterns(ArrayList<String>[] colKeyPatterns) {
		this.colKeyPatterns = colKeyPatterns;
	}

	public Types.ValueType[] getSchema() {
		return schema;
	}

	public void setSchema(Types.ValueType[] schema) {
		this.schema = schema;
	}

	public HashSet<String>[] endWithValueStrings() {
		if(endWithValueStrings !=null)
			return this.endWithValueStrings;
		else
			return null;
	}

	public ArrayList<String> getValueKeyPattern() {
		return valueKeyPattern;
	}

	public void setValueKeyPattern(ArrayList<String> valueKeyPattern) {
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

	public void setEndWithValueStrings(HashSet<String>[] endWithValueStrings) {
		this.endWithValueStrings = endWithValueStrings;
	}
}
