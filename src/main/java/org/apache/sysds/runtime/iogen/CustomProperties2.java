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
import org.apache.sysds.runtime.io.FileFormatProperties;

import java.io.Serializable;
import java.util.HashSet;

public class CustomProperties2 extends FileFormatProperties implements Serializable {
	private static final Log LOG = LogFactory.getLog(CustomProperties2.class.getName());
	private static final long serialVersionUID = -4447926749068752721L;

	public enum IndexProperties {
		IDENTIFY, PREFIX, KEY;
		@Override
		public String toString() {
			return this.name().toLowerCase().replaceAll("_", "-");
		}
	}

	private IndexProperties rowIndex;
	private IndexProperties colIndex;

	// When the index is prefixes
	private Integer rowIndexPrefixPosition;
	private String rowIndexPrefixDelim;
	private Boolean rowIndexPrefixDelimFixLength;

	public IndexProperties getRowIndex() {
		return rowIndex;
	}

	public void setRowIndex(IndexProperties rowIndex) {
		this.rowIndex = rowIndex;
	}

	public IndexProperties getColIndex() {
		return colIndex;
	}

	public void setColIndex(IndexProperties colIndex) {
		this.colIndex = colIndex;
	}

	public Integer getRowIndexPrefixPosition() {
		return rowIndexPrefixPosition;
	}

	public void setRowIndexPrefixPosition(Integer rowIndexPrefixPosition) {
		this.rowIndexPrefixPosition = rowIndexPrefixPosition;
	}

	public String getRowIndexPrefixDelim() {
		return rowIndexPrefixDelim;
	}

	public void setRowIndexPrefixDelim(String rowIndexPrefixDelim) {
		this.rowIndexPrefixDelim = rowIndexPrefixDelim;
	}

	public Boolean getRowIndexPrefixDelimFixLength() {
		return rowIndexPrefixDelimFixLength;
	}

	public void setRowIndexPrefixDelimFixLength(Boolean rowIndexPrefixDelimFixLength) {
		this.rowIndexPrefixDelimFixLength = rowIndexPrefixDelimFixLength;
	}
}
