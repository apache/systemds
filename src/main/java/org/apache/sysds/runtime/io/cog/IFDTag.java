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

package org.apache.sysds.runtime.io.cog;

/**
 * Represents a single tag in the IFD of a TIFF file
 */
public class IFDTag {
	private IFDTagDictionary tagId;
	private short dataType;
	private int dataCount;
	private Number[] data;

	public IFDTag(IFDTagDictionary tagId, short dataType, int dataCount, Number[] data) {
		this.tagId = tagId;
		this.dataType = dataType;
		this.dataCount = dataCount;
		this.data = data;
	}

	public IFDTagDictionary getTagId() {
		return tagId;
	}

	public void setTagId(IFDTagDictionary tagId) {
		this.tagId = tagId;
	}

	public short getDataType() {
		return dataType;
	}

	public void setDataType(short dataType) {
		this.dataType = dataType;
	}

	public int getDataCount() {
		return dataCount;
	}

	public void setDataCount(int dataCount) {
		this.dataCount = dataCount;
	}

	public Number[] getData() {
		return data;
	}

	public void setData(Number[] data) {
		this.data = data;
	}
}
