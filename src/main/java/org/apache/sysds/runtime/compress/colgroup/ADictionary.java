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
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * This dictionary class aims to encapsulate the storage and operations over unique floating point values of a column
 * group.
 */
public abstract class ADictionary {

	// private static final Log LOG = LogFactory.getLog(ADictionary.class.getName());

	/**
	 * Get all the values contained in the dictionary as a linearized double array.
	 * 
	 * @return linearized double array
	 */
	public abstract double[] getValues();

	/**
	 * Get Specific value contained in the dictionary at index.
	 * 
	 * @param i The index to extract the value from
	 * @return The value contained at the index
	 */
	public abstract double getValue(int i);

	/**
	 * Determines if the content has a zero tuple. meaning all values at a specific row are zero value. This is useful
	 * information to find out if the dictionary is used in a dense context. To improve some specific operations.
	 * 
	 * @param nCol The number of columns in the dictionary.
	 * @return The index at which the zero tuple is located.
	 */
	public abstract int hasZeroTuple(int nCol);

	/**
	 * Returns the memory usage of the dictionary.
	 * 
	 * @return a long value in number of bytes for the dictionary.
	 */
	public abstract long getInMemorySize();

	/**
	 * Aggregate all the contained values, useful in value only computations where the operation is iterating through
	 * all values contained in the dictionary.
	 * 
	 * @param init The initial Value, in cases such as Max value, this could be -infinity
	 * @param fn   The Function to apply to values
	 * @return The aggregated value as a double.
	 */
	public abstract double aggregate(double init, Builtin fn);

	/**
	 * Aggregate all entries in the rows.
	 * 
	 * @param fn The aggregate function
	 * @param nCol The number of columns contained in the dictionary.
	 * @return Aggregates for this dictionary tuples.
	 */
	public abstract double[] aggregateTuples(Builtin fn, int nCol);

	/**
	 * returns the count of values contained in the dictionary.
	 * 
	 * @return an integer of count of values.
	 */
	public abstract int size();

	/**
	 * Applies the scalar operation on the dictionary. Note that this operation modifies the underlying data, and
	 * normally require a copy of the original Dictionary to preserve old objects.
	 * 
	 * @param op The operator to apply to the dictionary values.
	 * @return this dictionary with modified values.
	 */
	public abstract ADictionary apply(ScalarOperator op);

	/**
	 * Applies the scalar operation on the dictionary. The returned dictionary should contain a new instance of the
	 * underlying data. Therefore it will not modify the previous object.
	 * 
	 * @param op      The operator to apply to the dictionary values.
	 * @param newVal  The value to append to the dictionary.
	 * @param numCols The number of columns stored in the dictionary.
	 * @return Another dictionary with modified values.
	 */
	public abstract ADictionary applyScalarOp(ScalarOperator op, double newVal, int numCols);

	public ADictionary applyBinaryRowOp(ValueFunction fn, double[] v, boolean sparseSafe, int[] colIndexes,
		boolean left) {
		return (left) ? applyBinaryRowOpLeft(fn, v, sparseSafe, colIndexes) : applyBinaryRowOpRight(fn,
			v,
			sparseSafe,
			colIndexes);
	}

	public abstract ADictionary applyBinaryRowOpLeft(ValueFunction fn, double[] v, boolean sparseSafe,
		int[] colIndexes);

	public abstract ADictionary applyBinaryRowOpRight(ValueFunction fn, double[] v, boolean sparseSafe,
		int[] colIndexes);

	/**
	 * Returns a deep clone of the dictionary.
	 */
	public abstract ADictionary clone();

	public abstract ADictionary cloneAndExtend(int len);

	/**
	 * Aggregates the columns into the target double array provided.
	 * 
	 * @param c          The target double array, this contains the full number of columns, therefore the colIndexes for
	 *                   this specific dictionary is needed.
	 * @param fn         The function to apply to individual columns
	 * @param colIndexes The mapping to the target columns from the individual columns
	 */
	public void aggregateCols(double[] c, Builtin fn, int[] colIndexes) {
		int ncol = colIndexes.length;
		int vlen = size() / ncol;
		for(int k = 0; k < vlen; k++)
			for(int j = 0, valOff = k * ncol; j < ncol; j++)
				c[colIndexes[j]] = fn.execute(c[colIndexes[j]], getValue(valOff + j));
	}

	/**
	 * The read function to instantiate the dictionary.
	 * 
	 * @param in    The data input source to read the stored dictionary from
	 * @param lossy Boolean specifying if the dictionary stored was lossy.
	 * @return The concrete dictionary.
	 * @throws IOException if the reading source throws it.
	 */
	public static ADictionary read(DataInput in, boolean lossy) throws IOException {
		return lossy ? QDictionary.read(in) : Dictionary.read(in);
	}

	/**
	 * Write the dictionary to a DataOutput.
	 * 
	 * @param out the output sink to write the dictionary to.
	 * @throws IOException if the sink fails.
	 */
	public abstract void write(DataOutput out) throws IOException;

	/**
	 * Calculate the space consumption if the dictionary is stored on disk.
	 * 
	 * @return the long count of bytes to store the dictionary.
	 */
	public abstract long getExactSizeOnDisk();

	/**
	 * Get the number of values given that the column group has n columns
	 * 
	 * @param ncol The number of Columns in the ColumnGroup.
	 * @return the number of value tuples contained in the dictionary.
	 */
	public abstract int getNumberOfValues(int ncol);

	/**
	 * Materializes a Zero tuple at the last index of the dictionary.
	 * 
	 * @param numCols The number of columns in the dictionary
	 * @return the new Dictionary with materialized zero tuple.
	 */
	// public abstract IDictionary materializeZeroValue(int numCols);

	/**
	 * Method used as a pre-aggregate of each tuple in the dictionary, to single double values.
	 * 
	 * Note if the number of columns is one the actual dictionaries values are simply returned.
	 * 
	 * @param square    If each entry should be squared.
	 * @param nrColumns The number of columns in the ColGroup to know how to get the values from the dictionary.
	 * @return a double array containing the row sums from this dictionary.
	 */
	protected abstract double[] sumAllRowsToDouble(boolean square, int nrColumns);

	/**
	 * Sum the values at a specific row.
	 * 
	 * @param k         The row index to sum
	 * @param square    If each entry should be squared.
	 * @param nrColumns The number of columns
	 * @return The sum of the row.
	 */
	protected abstract double sumRow(int k, boolean square, int nrColumns);

	protected abstract void colSum(double[] c, int[] counts, int[] colIndexes, boolean square);

	protected abstract double sum(int[] counts, int ncol);

	protected abstract double sumsq(int[] counts, int ncol);

	public abstract StringBuilder getString(StringBuilder sb, int colIndexes);

	/**
	 * This method adds the max and min values contained in the dictionary to corresponding cells in the ret variable.
	 * 
	 * One use case for this method is the squash operation, to go from an overlapping state to normal compression.
	 * 
	 * @param ret        The double array that contains all columns min and max.
	 * @param colIndexes The column indexes contained in this dictionary.
	 */
	protected abstract void addMaxAndMin(double[] ret, int[] colIndexes);

	/**
	 * Modify the dictionary by removing columns not within the index range.
	 * 
	 * @param idxStart                The column index to start at.
	 * @param idxEnd                  The column index to end at (not inclusive)
	 * @param previousNumberOfColumns The number of columns contained in the dictionary.
	 * @return A dictionary containing the sliced out columns values only.
	 */
	public abstract ADictionary sliceOutColumnRange(int idxStart, int idxEnd, int previousNumberOfColumns);

	/**
	 * return a new Dictionary that have re expanded all values, based on the entries already contained.
	 * 
	 * @param max The number of output columns possible.
	 * @return The re expanded Dictionary.
	 */
	public abstract ADictionary reExpandColumns(int max);

	public abstract boolean containsValue(double pattern);

	public abstract long getNumberNonZeros(int[] counts, int nCol);
}
