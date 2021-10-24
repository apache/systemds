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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.io.FileFormatProperties;

import java.io.Serializable;
import java.util.HashSet;

public class CustomProperties extends FileFormatProperties implements Serializable {
	protected static final Log LOG = LogFactory.getLog(CustomProperties.class.getName());
	private static final long serialVersionUID = -4447926749068752721L;

	private String delim;
	private String indexDelim;
	private HashSet<String> naStrings;
	private int firstColIndex;
	private int firstRowIndex;
	private String[] colKeys;
	private Types.ValueType[] schema;

	protected enum GRPattern {
		Regular, Irregular;

		@Override
		public String toString() {
			return this.name().toLowerCase();
		}
	}

	protected enum GRSymmetry {
		GENERAL, SYMMETRIC, SKEW_SYMMETRIC;

		@Override
		public String toString() {
			return this.name().toLowerCase().replaceAll("_", "-");
		}
	}

	private GRPattern rowPattern;
	private GRPattern colPattern;
	private GRSymmetry grSymmetry;

	public CustomProperties() {
	}

	// Row & Col Regular Format
	public CustomProperties(GRPattern rowPattern, String delim, HashSet<String> naStrings) {
		this.delim = delim;
		this.naStrings = naStrings;
		this.rowPattern = rowPattern;
		this.colPattern = GRPattern.Regular;
		this.grSymmetry = GRSymmetry.GENERAL;
		this.firstRowIndex = 0;
		this.firstColIndex = 0;
	}

	// Row Regular & Col Irregular Format
	public CustomProperties(GRPattern rowPattern, String delim, String indexDelim, int firstColIndex) {
		this.delim = delim;
		this.indexDelim = indexDelim;
		this.rowPattern = rowPattern;
		this.colPattern = GRPattern.Irregular;
		this.grSymmetry = GRSymmetry.GENERAL;
		this.firstColIndex = firstColIndex;
		this.firstRowIndex = 0;
	}

	// Row Irregular format
	public CustomProperties(GRSymmetry grSymmetry, String delim, int firstRowIndex, int firstColIndex) {
		this.delim = delim;
		this.grSymmetry = grSymmetry;
		this.colPattern = GRPattern.Regular;
		this.rowPattern = GRPattern.Irregular;
		this.firstColIndex = firstColIndex;
		this.firstRowIndex = firstRowIndex;
	}

	// Nested format
	public CustomProperties(String[] colKeys, Types.ValueType[] schema) {
		this.colKeys = colKeys;
		this.schema = schema;
		this.rowPattern = GRPattern.Regular;
	}

	public CustomProperties(String[] colKeys) {
		this.colKeys = colKeys;

	}

	public String getDelim() {
		return delim;
	}

	public String getIndexDelim() {
		return indexDelim;
	}

	public HashSet<String> getNaStrings() {
		return naStrings;
	}

	public GRPattern getRowPattern() {
		return rowPattern;
	}

	public GRPattern getColPattern() {
		return colPattern;
	}

	public GRSymmetry getGrSymmetry() {
		return grSymmetry;
	}

	public int getFirstColIndex() {
		return firstColIndex;
	}

	public void setFirstColIndex(int firstColIndex) {
		this.firstColIndex = firstColIndex;
	}

	public int getFirstRowIndex() {
		return firstRowIndex;
	}

	public void setFirstRowIndex(int firstRowIndex) {
		this.firstRowIndex = firstRowIndex;
	}

	public String[] getColKeys() {
		return colKeys;
	}

	public Types.ValueType[] getSchema() {
		return schema;
	}
}
