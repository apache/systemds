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

public class ColIndexStructure {

	public enum IndexProperties {
		Identity, // col number of sample raw data equal to the row index of matrix/frame
		CellWiseExist; // col index of every cell values are in the sample raw data
		@Override
		public String toString() {
			return this.name().toUpperCase();
		}
	}

	public ColIndexStructure() {
		this.properties = null;
		this.keyPattern = null;
		this.colIndexBegin = "0";
	}

	private IndexProperties properties;
	private KeyTrie keyPattern;
	private String colIndexBegin;

	// when the index properties is CellWiseExist:
	private String indexDelim;
	private String valueDelim;

	public IndexProperties getProperties() {
		return properties;
	}

	public void setProperties(IndexProperties properties) {
		this.properties = properties;
	}

	public KeyTrie getKeyPattern() {
		return keyPattern;
	}

	public void setKeyPattern(KeyTrie keyPattern) {
		this.keyPattern = keyPattern;
	}

	public String getColIndexBegin() {
		return colIndexBegin;
	}

	public void setColIndexBegin(int colIndexBegin) {
		this.colIndexBegin = colIndexBegin + "";
	}

	public String getIndexDelim() {
		return indexDelim;
	}

	public void setIndexDelim(String indexDelim) {
		this.indexDelim = indexDelim;
	}

	public String getValueDelim() {
		return valueDelim;
	}

	public void setValueDelim(String valueDelim) {
		this.valueDelim = valueDelim;
	}
}
