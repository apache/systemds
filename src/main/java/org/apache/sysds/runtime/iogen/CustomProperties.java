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

public class CustomProperties extends FileFormatProperties implements Serializable {

	public enum IndexProperties {
		IDENTIFY, PREFIX, KEY;
		@Override public String toString() {
			return this.name().toUpperCase();
		}
	}

	private KeyTrie[] colKeyPattern;
	private Types.ValueType[] schema;
	private IndexProperties rowIndex;
	private KeyTrie rowKeyPattern;

	public CustomProperties(KeyTrie[] colKeyPattern, IndexProperties rowIndex) {
		this.colKeyPattern = colKeyPattern;
		this.rowIndex = rowIndex;
	}

	public CustomProperties(KeyTrie[] colKeyPattern, IndexProperties rowIndex, KeyTrie rowKeyPattern) {
		this.colKeyPattern = colKeyPattern;
		this.rowIndex = rowIndex;
		this.rowKeyPattern = rowKeyPattern;
	}

	public KeyTrie[] getColKeyPattern() {
		return colKeyPattern;
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
}
