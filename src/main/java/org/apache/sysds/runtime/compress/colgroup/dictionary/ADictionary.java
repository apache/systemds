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

import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * This dictionary class aims to encapsulate the storage and operations over unique tuple values of a column group.
 */
public abstract class ADictionary implements Serializable {

	private static final long serialVersionUID = 9118692576356558592L;

	protected static final Log LOG = LogFactory.getLog(ADictionary.class.getName());

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
	 * Returns the memory usage of the dictionary.
	 * 
	 * @return a long value in number of bytes for the dictionary.
	 */
	public abstract long getInMemorySize();

	/**
	 * Aggregate all the contained values, useful in value only computations where the operation is iterating through all
	 * values contained in the dictionary.
	 * 
	 * @param init The initial Value, in cases such as Max value, this could be -infinity
	 * @param fn   The Function to apply to values
	 * @return The aggregated value as a double.
	 */
	public abstract double aggregate(double init, Builtin fn);

	/**
	 * Aggregate all entries in the rows.
	 * 
	 * @param fn   The aggregate function
	 * @param nCol The number of columns contained in the dictionary.
	 * @return Aggregates for this dictionary tuples.
	 */
	public abstract double[] aggregateTuples(Builtin fn, int nCol);

	/**
	 * Applies the scalar operation on the dictionary. Note that this operation modifies the underlying data, and
	 * normally require a copy of the original Dictionary to preserve old objects.
	 * 
	 * @param op The operator to apply to the dictionary values.
	 * @return this dictionary with modified values.
	 */
	public abstract ADictionary inplaceScalarOp(ScalarOperator op);

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

	/**
	 * Apply binary row operation on the left side in place
	 * 
	 * @param op         The operation to this dictionary
	 * @param v          The values to use on the left hand side.
	 * @param colIndexes The column indexes to consider inside v.
	 * @return A new dictionary containing the updated values.
	 */
	public abstract ADictionary binOpLeft(BinaryOperator op, double[] v, int[] colIndexes);

	/**
	 * Apply binary row operation on the right side.
	 * 
	 * @param op         The operation to this dictionary
	 * @param v          The values to use on the right hand side.
	 * @param colIndexes The column indexes to consider inside v.
	 * @return A new dictionary containing the updated values.
	 */
	public abstract ADictionary binOpRight(BinaryOperator op, double[] v, int[] colIndexes);

	/**
	 * Apply binary row operation on the left side and allocate a new dictionary.
	 * 
	 * While adding a new tuple, where the operation is applied with zero values.
	 * 
	 * @param op         The operation to this dictionary
	 * @param v          The values to use on the left hand side.
	 * @param colIndexes The column indexes to consider inside v.
	 * @return A new dictionary containing the updated values.
	 */
	public abstract ADictionary applyBinaryRowOpLeftAppendNewEntry(BinaryOperator op, double[] v, int[] colIndexes);


	/**
	 * Apply binary row operation on this dictionary on the right side.
	 * 
	 * @param op         The operation to this dictionary
	 * @param v          The values to use on the right hand side.
	 * @param colIndexes The column indexes to consider inside v.
	 * @return A new dictionary containing the updated values.
	 */
	public abstract ADictionary applyBinaryRowOpRightAppendNewEntry(BinaryOperator op, double[] v, int[] colIndexes);

	/**
	 * Returns a deep clone of the dictionary.
	 */
	public abstract ADictionary clone();

	/**
	 * Clone the dictionary, and extend size of the dictionary by a given length
	 * 
	 * @param len The length to extend the dictionary, it is assumed this value is positive.
	 * @return a clone of the dictionary, extended by len.
	 */
	public abstract ADictionary cloneAndExtend(int len);

	/**
	 * Aggregates the columns into the target double array provided.
	 * 
	 * @param c          The target double array, this contains the full number of columns, therefore the colIndexes for
	 *                   this specific dictionary is needed.
	 * @param fn         The function to apply to individual columns
	 * @param colIndexes The mapping to the target columns from the individual columns
	 */
	public abstract void aggregateCols(double[] c, Builtin fn, int[] colIndexes);

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
	 * Specify if the Dictionary is lossy.
	 * 
	 * @return A boolean
	 */
	public abstract boolean isLossy();

	/**
	 * Get the number of distinct tuples given that the column group has n columns
	 * 
	 * @param ncol The number of Columns in the ColumnGroup.
	 * @return the number of value tuples contained in the dictionary.
	 */
	public abstract int getNumberOfValues(int ncol);

	/**
	 * Method used as a pre-aggregate of each tuple in the dictionary, to single double values.
	 * 
	 * Note if the number of columns is one the actual dictionaries values are simply returned.
	 * 
	 * @param square    If each entry should be squared.
	 * @param nrColumns The number of columns in the ColGroup to know how to get the values from the dictionary.
	 * @return a double array containing the row sums from this dictionary.
	 */
	public abstract double[] sumAllRowsToDouble(boolean square, int nrColumns);

	/**
	 * Sum the values at a specific row.
	 * 
	 * @param k         The row index to sum
	 * @param square    If each entry should be squared.
	 * @param nrColumns The number of columns
	 * @return The sum of the row.
	 */
	public abstract double sumRow(int k, boolean square, int nrColumns);

	/**
	 * get the column sum of this dictionary only.
	 * 
	 * @param counts the counts of the values contained
	 * @param nCol   The number of columns contained in each tuple.
	 * @return the colSums of this column group.
	 */
	public abstract double[] colSum(int[] counts, int nCol);

	/**
	 * Get the column sum of the values contained in the dictionary
	 * 
	 * @param c          The output array allocated to contain all column groups output.
	 * @param counts     The counts of the individual tuples.
	 * @param colIndexes The columns indexes of the parent column group, this indicate where to put the column sum into
	 *                   the c output.
	 * @param square     Specify if the values should be squared
	 */
	public abstract void colSum(double[] c, int[] counts, int[] colIndexes, boolean square);

	/**
	 * Get the sum of the values contained in the dictionary
	 * 
	 * @param counts The counts of the individual tuples
	 * @param nCol   The number of columns contained
	 * @return The sum scaled by the counts provided.
	 */
	public abstract double sum(int[] counts, int nCol);

	/**
	 * Get the square sum of the values contained in the dictionary
	 * 
	 * @param counts The counts of the individual tuples
	 * @param nCol   The number of columns contained
	 * @return The square sum scaled by the counts provided.
	 */
	public abstract double sumsq(int[] counts, int nCol);

	/**
	 * Get a string representation of the dictionary, that considers the layout of the data.
	 * 
	 * @param colIndexes The number of columns in the dictionary.
	 * @return A string that is nicer to print.
	 */
	public abstract String getString(int colIndexes);

	/**
	 * This method adds the max and min values contained in the dictionary to corresponding cells in the ret variable.
	 * 
	 * One use case for this method is the squash operation, to go from an overlapping state to normal compression.
	 * 
	 * @param ret        The double array that contains all columns min and max.
	 * @param colIndexes The column indexes contained in this dictionary.
	 */
	public abstract void addMaxAndMin(double[] ret, int[] colIndexes);

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

	/**
	 * Detect if the dictionary contains a specific value.
	 * 
	 * @param pattern The value to search for
	 * @return true if the value is contained else false.
	 */
	public abstract boolean containsValue(double pattern);

	/**
	 * Calculate the number of non zeros in the dictionary. The number of non zeros should be scaled with the counts
	 * given. This gives the exact number of non zero values in the parent column group.
	 * 
	 * @param counts The counts of each dictionary entry
	 * @param nCol   The number of columns in this dictionary
	 * @return The nonZero count
	 */
	public abstract long getNumberNonZeros(int[] counts, int nCol);

	/**
	 * Copies and adds the dictionary entry from this dictionary to the d dictionary
	 * 
	 * @param d    the target dictionary
	 * @param fr   the from index
	 * @param to   the to index
	 * @param nCol the number of columns
	 */
	public abstract void addToEntry(Dictionary d, int fr, int to, int nCol);

	/**
	 * Get the values contained in a specific tuple of the dictionary.
	 * 
	 * If the entire row is zero return null.
	 * 
	 * @param index The index where the values are located
	 * @param nCol  The number of columns contained in this dictionary
	 * @return a materialized double array containing the tuple.
	 */
	public abstract double[] getTuple(int index, int nCol);

	/**
	 * Allocate a new dictionary where the tuple given is subtracted from all tuples in the previous dictionary.
	 * 
	 * @param tuple a double list representing a tuple, it is given that the tuple with is the same as this dictionaries.
	 * @return a new instance of dictionary with the tuple subtracted.
	 */
	public abstract ADictionary subtractTuple(double[] tuple);

	/**
	 * Get this dictionary as a matrixBlock dictionary. This allows us to use optimized kernels coded elsewhere in the
	 * system, such as matrix multiplication.
	 * 
	 * @param nCol The number of columns contained in this column group.
	 * @return A Dictionary containing a MatrixBlock.
	 */
	public abstract MatrixBlockDictionary getMBDict(int nCol);

	/**
	 * Scale all tuples contained in the dictionary by the scaling factor given in the int list.
	 * 
	 * @param scaling The ammout to multiply the given tuples with
	 * @param nCol    The number of columns contained in this column group.
	 * @return A New dictionary (since we don't want to modify the underlying dictionary)
	 */
	public abstract ADictionary scaleTuples(int[] scaling, int nCol);

	/**
	 * Pre Aggregate values for right Matrix Multiplication.
	 * 
	 * @param numVals          The number of values contained in this dictionary
	 * @param colIndexes       The column indexes that is associated with the parent column group
	 * @param aggregateColumns The column to aggregate, this is preprocessed, to find remove consideration for empty
	 *                         columns
	 * @param b                The values in the right hand side matrix
	 * @param cut              The number of columns in b.
	 * @return A new dictionary with the pre aggregated values.
	 */
	public abstract ADictionary preaggValuesFromDense(final int numVals, final int[] colIndexes,
		final int[] aggregateColumns, final double[] b, final int cut);

	/**
	 * Make a copy of the values, and replace all values that match pattern with replacement value. If needed add a new
	 * column index.
	 * 
	 * @param pattern The value to look for
	 * @param replace The value to replace the other value with
	 * @param nCol    The number of columns contained in the dictionary.
	 * @return A new Column Group, reusing the index structure but with new values.
	 */
	public abstract ADictionary replace(double pattern, double replace, int nCol);

	public abstract ADictionary replaceZeroAndExtend(double replace, int nCol);

	public abstract double product(int[] counts, int nCol);

	public abstract void colProduct(double[] res, int[] counts, int[] colIndexes);
}
