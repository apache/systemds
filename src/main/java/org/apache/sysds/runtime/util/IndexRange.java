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

package org.apache.sysds.runtime.util;

import java.io.Serializable;

//start and end are all inclusive
public class IndexRange implements Serializable
{
	private static final long serialVersionUID = 5746526303666494601L;
	
	public long rowStart = 0;
	public long rowEnd = 0;
	public long colStart = 0;
	public long colEnd = 0;

	public static IndexRange intersect(IndexRange a, IndexRange b) {
		return new IndexRange(Math.max(a.rowStart, b.rowStart), Math.min(a.rowEnd, b.rowEnd), Math.max(a.colStart, b.colStart), Math.min(a.colEnd, b.colEnd));
	}
	
	public IndexRange(long rs, long re, long cs, long ce) {
		set(rs, re, cs, ce);
	}
	
	public void set(long rs, long re, long cs, long ce) {
		rowStart = rs;
		rowEnd = re;
		colStart = cs;
		colEnd = ce;
	}
	
	public boolean isScalar() {
		return (rowStart==rowEnd && colStart==colEnd);
	}
	
	public IndexRange add(int delta) {
		return new IndexRange(
			rowStart + delta, rowEnd + delta,
			colStart + delta, colEnd + delta);
	}

	public IndexRange add(long rowDelta, long colDelta) {
		return new IndexRange(rowStart + rowDelta, rowEnd + rowDelta, colStart + colDelta, colEnd + colDelta);
	}

	public boolean inColRange(long col) {
		return col >= colStart && col < colEnd;
	}

	public boolean inRowRange(long row) {
		return row >= rowStart && row < rowEnd;
	}

	public long colSpan() {
		return colEnd - colStart;
	}

	public long rowSpan() {
		return rowEnd - rowStart;
	}

	public boolean isWithin(long row, long col) {
		return inColRange(col) && inRowRange(row);
	}

	@Override
	public String toString() {
		return "["+rowStart+":"+rowEnd+","+colStart+":"+colEnd+"]";
	}
}
