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

package org.apache.sysds.runtime.compress.colgroup;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;


/**
 * This dictionary class aims to encapsulate the storage and operations over unique floating point values of a column
 * group. The primary reason for its introduction was to provide an entry point for specialization such as shared
 * dictionaries, which require additional information.
 */
public abstract class IDictionary {

	public abstract double[] getValues();

	public abstract double getValue(int i);

	public abstract int hasZeroTuple(int ncol);

	public abstract long getInMemorySize();

	public abstract double aggregate(double init, Builtin fn);

	public abstract int getValuesLength();

	public abstract IDictionary apply(ScalarOperator op);

	public abstract IDictionary clone();

	public double[] aggregateCols(double[] init, Builtin fn, int[] cols) {
		int ncol = cols.length;
		int vlen = getValuesLength() / ncol;
		double[] ret = init;
		for(int k = 0; k < vlen; k++)
			for(int j = 0, valOff = k * ncol; j < ncol; j++)
				ret[j] = fn.execute(ret[j], getValue(valOff + j));
		return ret;
	}

	public static IDictionary read(DataInput in, boolean lossy) throws IOException {
		return lossy ? QDictionary.read(in) : Dictionary.read(in);
	}

	public abstract void write(DataOutput out) throws IOException;

	public abstract long getExactSizeOnDisk();

	/**
	 * Get the number of values given that the column group has n columns
	 * @param ncol The number of Columns in the ColumnGroup.
	 */
	public abstract int getNumberOfValues(int ncol);

	public static IDictionary materializeZeroValue(IDictionary OldDictionary, int numCols){
		if(OldDictionary instanceof QDictionary){
			return QDictionary.materializeZeroValueLossy((QDictionary)OldDictionary, numCols);
		} else{
			return Dictionary.materializeZeroValueFull((Dictionary)OldDictionary, numCols);
		}
	}

	protected abstract double[] sumAllRowsToDouble(KahanFunction kplus, KahanObject kbuff, int nrColumns,  boolean allocNew);

	protected abstract double sumRow(int k, KahanFunction kplus, KahanObject kbuff, int nrColumns);
}
