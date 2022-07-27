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

import java.util.HashSet;

public class RowIndexStructure {

	public enum IndexProperties {
		Identity, // line number of sample raw data equal to the row index of matrix/frame
		CellWiseExist, // row index of every cell values are in the sample raw data
		RowWiseExist, // index of every record in matrix/frame has an index in sample raw
		SeqScatter; // the row index is not exist but the record scattered sequentially in multi lines
		@Override
		public String toString() {
			return this.name().toUpperCase();
		}
	}

	public RowIndexStructure() {
		this.properties = null;
		this.keyPattern = null;
		this.rowIndexBegin = 0;
	}

	private IndexProperties properties;
	private KeyTrie keyPattern;
	private int rowIndexBegin;

	private String seqBeginString;
	private String seqEndString;

	public HashSet<String> endWithValueStrings() {
		if(keyPattern!=null) {
			HashSet<String> endWithValueString = keyPattern.getFirstSuffixKeyPatterns();
			return endWithValueString;
		}
		else
			return null;
	}

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

	public int getRowIndexBegin() {
		return rowIndexBegin;
	}

	public void setRowIndexBegin(int rowIndexBegin) {
		this.rowIndexBegin = rowIndexBegin;
	}

	public String getSeqBeginString() {
		return seqBeginString;
	}

	public void setSeqBeginString(String seqBeginString) {
		this.seqBeginString = seqBeginString;
	}

	public String getSeqEndString() {
		return seqEndString;
	}

	public void setSeqEndString(String seqEndString) {
		this.seqEndString = seqEndString;
	}
}
