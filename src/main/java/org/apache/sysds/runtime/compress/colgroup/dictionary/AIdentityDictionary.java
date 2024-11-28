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

package org.apache.sysds.runtime.compress.colgroup.dictionary;

import org.apache.sysds.runtime.compress.DMLCompressionException;

public abstract class AIdentityDictionary extends ACachingMBDictionary {
	private static final long serialVersionUID = 5013713435287705877L;
	/** The number of rows or columns, rows can be +1 if withEmpty is set. */
	protected final int nRowCol;
	/** Specify if the Identity matrix should contain an empty row in the end. */
	protected final boolean withEmpty;

	/**
	 * Create an identity matrix dictionary. It behaves as if allocated a Sparse Matrix block but exploits that the
	 * structure is known to have certain properties.
	 * 
	 * @param nRowCol The number of rows and columns in this identity matrix.
	 */
	public AIdentityDictionary(int nRowCol) {
		if(nRowCol <= 0)
			throw new DMLCompressionException("Invalid Identity Dictionary");
		this.nRowCol = nRowCol;
		this.withEmpty = false;
	}

	public AIdentityDictionary(int nRowCol, boolean withEmpty) {
		if(nRowCol <= 0)
			throw new DMLCompressionException("Invalid Identity Dictionary");
		this.nRowCol = nRowCol;
		this.withEmpty = withEmpty;
	}

	public boolean withEmpty() {
		return withEmpty;
	}

	public static long getInMemorySize(int numberColumns) {
		return 4 + 4 + 8; // int + padding + softReference
	}

	@Override
	public final boolean containsValue(double pattern) {
		return pattern == 0.0 || pattern == 1.0;
	}

	@Override
	public double[] productAllRowsToDouble(int nCol) {
		return new double[nRowCol + (withEmpty ? 1 : 0)];
	}

	@Override
	public double[] productAllRowsToDoubleWithDefault(double[] defaultTuple) {
		double[] ret = new double[nRowCol + (withEmpty ? 1 : 0) + 1];
		ret[ret.length - 1] = 1;
		for(int i = 0; i < defaultTuple.length; i++)
			ret[ret.length - 1] *= defaultTuple[i];
		return ret;
	}
}
