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

package org.apache.sysds.test.functions.iogen.baseline;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.io.FileFormatProperties;

import java.io.Serializable;

public class FileFormatPropertiesHL7 extends FileFormatProperties implements Serializable
{
	protected static final Log LOG = LogFactory.getLog(FileFormatPropertiesHL7.class.getName());

	private int[] selectedIndexes;
	private int maxColumnIndex;

	private boolean readAllValues;
	private boolean rangeBaseRead;
	private boolean queryFilter;

	public FileFormatPropertiesHL7(int[] selectedIndexes, int maxColumnIndex) {
		this.selectedIndexes = selectedIndexes;
		this.maxColumnIndex = maxColumnIndex;

		if(this.maxColumnIndex == -1) {
			this.readAllValues = true;
			this.queryFilter = false;
			this.rangeBaseRead = false;
		}
		else {
			this.readAllValues = false;
			if(this.selectedIndexes.length > 0) {
				this.queryFilter = true;
				this.rangeBaseRead = false;
				this.maxColumnIndex = this.selectedIndexes[0];
				for(int i=1; i< this.selectedIndexes.length; i++)
					this.maxColumnIndex = Math.max(this.maxColumnIndex, this.selectedIndexes[i]);
			}
			else {
				this.rangeBaseRead = true;
				this.queryFilter = false;
			}
		}
	}

	public FileFormatPropertiesHL7() {
	}

	public FileFormatPropertiesHL7(int maxColumnIndex) {
		this.maxColumnIndex = maxColumnIndex;
	}

	public FileFormatPropertiesHL7(int[] selectedIndexes) {
		this.selectedIndexes = selectedIndexes;
	}

	public int[] getSelectedIndexes() {
		return selectedIndexes;
	}

	public void setSelectedIndexes(int[] selectedIndexes) {
		this.selectedIndexes = selectedIndexes;
	}

	public int getMaxColumnIndex() {
		return maxColumnIndex;
	}

	public void setMaxColumnIndex(int maxColumnIndex) {
		this.maxColumnIndex = maxColumnIndex;
	}

	public boolean isReadAllValues() {
		return readAllValues;
	}

	public void setReadAllValues(boolean readAllValues) {
		this.readAllValues = readAllValues;
	}

	public boolean isRangeBaseRead() {
		return rangeBaseRead;
	}

	public void setRangeBaseRead(boolean rangeBaseRead) {
		this.rangeBaseRead = rangeBaseRead;
	}

	public boolean isQueryFilter() {
		return queryFilter;
	}

	public void setQueryFilter(boolean queryFilter) {
		this.queryFilter = queryFilter;
	}
}
