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

package org.apache.sysml.runtime.compress;

import org.apache.sysml.runtime.compress.utils.DblArray;

/**
 * Base class for all column selection readers.
 * 
 */
public abstract class ReaderColumnSelection 
{
	protected int[] _colIndexes = null;
	protected int _numRows = -1;
	protected int _lastRow = -1;
	protected boolean _skipZeros = false;
	
	protected ReaderColumnSelection(int[] colIndexes, int numRows, boolean skipZeros) {
		_colIndexes = colIndexes;
		_numRows = numRows;
		_lastRow = -1;
		_skipZeros = skipZeros;
	}
	
	/**
	 * Gets the next row, null when no more rows.
	 * 
	 * @return
	 */
	public abstract DblArray nextRow();

	/**
	 * 
	 * @return
	 */
	public int getCurrentRowIndex() {
		return _lastRow;
	}
	

	/**
	 * Resets the reader to the first row.
	 */
	public void reset() {
		_lastRow = -1;
	}
}
