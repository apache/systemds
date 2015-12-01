/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */
package org.apache.sysml.runtime.instructions.spark.data;

import java.io.Serializable;

public class CountLinesInfo implements Serializable {
	private static final long serialVersionUID = 4178309746487858987L;
	private long numLines;
	private long expectedNumColumns;
	public long getNumLines() {
		return numLines;
	}
	public void setNumLines(long numLines) {
		this.numLines = numLines;
	}
	public long getExpectedNumColumns() {
		return expectedNumColumns;
	}
	public void setExpectedNumColumns(long expectedNumColumns) {
		this.expectedNumColumns = expectedNumColumns;
	}
	
}
