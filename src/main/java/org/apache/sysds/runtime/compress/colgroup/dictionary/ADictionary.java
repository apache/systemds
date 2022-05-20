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
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

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
	 * Get Specific value contain in dictionary at index.
	 * 
	 * @param r    Row target
	 * @param col  Col target
	 * @param nCol nCol in dictionary
	 * @return value
	 */
	public abstract double getValue(int r, int col, int nCol);

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
	 * Aggregate all the contained values, with a reference offset.
	 * 
	 * @param init      The initial value, in cases such as Max value this could be -infinity.
	 * @param fn        The function to apply to the values
	 * @param reference The reference offset to each value in the dictionary
	 * @param def       If the reference should be treated as an instance of only as reference
	 * @return The aggregated value as a double.
	 */
	public abstract double aggregateWithReference(double init, Builtin fn, double[] reference, boolean def);

	/**
	 * Aggregate all entries in the rows.
	 * 
	 * @param fn   The aggregate function
	 * @param nCol The number of columns contained in the dictionary.
	 * @return Aggregates for this dictionary tuples.
	 */
	public abstract double[] aggregateRows(Builtin fn, int nCol);

	/**
	 * Aggregate all entries in the rows of the dictionary with a extra cell in the end that contains the aggregate of
	 * the given defaultTuple.
	 * 
	 * @param fn           The aggregate function
	 * @param defaultTuple The default tuple to aggregate in last cell
	 * @return Aggregates for this dictionary tuples.
	 */
	public abstract double[] aggregateRowsWithDefault(Builtin fn, double[] defaultTuple);

	/**
	 * Aggregate all entries in the rows with an offset value reference added.
	 * 
	 * @param fn        The aggregate function
	 * @param reference The reference offset to each value in the dictionary
	 * @return Aggregates for this dictionary tuples.
	 */
	public abstract double[] aggregateRowsWithReference(Builtin fn, double[] reference);

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
	 * Aggregates the columns into the target double array provided.
	 * 
	 * @param c          The target double array, this contains the full number of columns, therefore the colIndexes for
	 *                   this specific dictionary is needed.
	 * @param fn         The function to apply to individual columns
	 * @param colIndexes The mapping to the target columns from the individual columns
	 * @param reference  The reference offset values to add to each cell.
	 * @param def        If the reference should be treated as a tuple as well
	 */
	public abstract void aggregateColsWithReference(double[] c, Builtin fn, int[] colIndexes, double[] reference,
		boolean def);

	/**
	 * Allocate a new dictionary and applies the scalar operation on each cell of to then return the new dictionary.
	 * 
	 * @param op The operator.
	 * @return The new dictionary to return.
	 */
	public abstract ADictionary applyScalarOp(ScalarOperator op);

	/**
	 * Allocate a new dictionary with one extra row and applies the scalar operation on each cell of to then return the
	 * new dictionary.
	 * 
	 * @param op   The operator
	 * @param v0   The new value to put into each cell in the new row
	 * @param nCol The number of columns in the dictionary
	 * @return The new dictionary to return.
	 */
	public abstract ADictionary applyScalarOpAndAppend(ScalarOperator op, double v0, int nCol);

	/**
	 * Allocate a new dictionary and apply the unary operator on each cell.
	 * 
	 * @param op The operator.
	 * @return The new dictionary to return.
	 */
	public abstract ADictionary applyUnaryOp(UnaryOperator op);

	/**
	 * Allocate a new dictionary with one extra row and apply the unary operator on each cell.
	 * 
	 * @param op   The operator.
	 * @param v0   The new value to put into each cell in the new row
	 * @param nCol The number of columns in the dictionary
	 * @return The new dictionary to return.
	 */
	public abstract ADictionary applyUnaryOpAndAppend(UnaryOperator op, double v0, int nCol);

	/**
	 * Allocate a new dictionary and apply the scalar operation on each cell to then return a new dictionary.
	 * 
	 * outValues[j] = op(this.values[j] + reference[i]) - newReference[i]
	 * 
	 * @param op           The operator to apply to each cell.
	 * @param reference    The reference value to add before the operator.
	 * @param newReference The reference value to subtract after the operator.
	 * @return A New Dictionary.
	 */
	public abstract ADictionary applyScalarOpWithReference(ScalarOperator op, double[] reference, double[] newReference);

	/**
	 * Allocate a new dictionary and apply the scalar operation on each cell to then return a new dictionary.
	 * 
	 * outValues[j] = op(this.values[j] + reference[i]) - newReference[i]
	 * 
	 * @param op           The unary operator to apply to each cell.
	 * @param reference    The reference value to add before the operator.
	 * @param newReference The reference value to subtract after the operator.
	 * @return A New Dictionary.
	 */
	public abstract ADictionary applyUnaryOpWithReference(UnaryOperator op, double[] reference, double[] newReference);

	/**
	 * Apply binary row operation on the left side
	 * 
	 * @param op         The operation to this dictionary
	 * @param v          The values to use on the left hand side.
	 * @param colIndexes The column indexes to consider inside v.
	 * @return A new dictionary containing the updated values.
	 */
	public abstract ADictionary binOpLeft(BinaryOperator op, double[] v, int[] colIndexes);

	/**
	 * Apply binary row operation on the left side with one extra row evaluating with zeros.
	 * 
	 * @param op         The operation to this dictionary
	 * @param v          The values to use on the left hand side.
	 * @param colIndexes The column indexes to consider inside v.
	 * @return A new dictionary containing the updated values.
	 */
	public abstract ADictionary binOpLeftAndAppend(BinaryOperator op, double[] v, int[] colIndexes);

	/**
	 * Apply the binary operator such that each value is offset by the reference before application. Then put the result
	 * into the new dictionary, but offset it by the new reference.
	 * 
	 * outValues[j] = op(v[colIndexes[i]], this.values[j] + reference[i]) - newReference[i]
	 * 
	 * 
	 * @param op           The operation to apply on the dictionary values.
	 * @param v            The values to use on the left side of the operator.
	 * @param colIndexes   The column indexes to use.
	 * @param reference    The reference value to add before operator.
	 * @param newReference The reference value to subtract after operator.
	 * @return A new dictionary.
	 */
	public abstract ADictionary binOpLeftWithReference(BinaryOperator op, double[] v, int[] colIndexes,
		double[] reference, double[] newReference);

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
	 * Apply binary row operation on the right side with one extra row evaluating with zeros.
	 * 
	 * @param op         The operation to this dictionary
	 * @param v          The values to use on the right hand side.
	 * @param colIndexes The column indexes to consider inside v.
	 * @return A new dictionary containing the updated values.
	 */
	public abstract ADictionary binOpRightAndAppend(BinaryOperator op, double[] v, int[] colIndexes);

	/**
	 * Apply binary row operation on the right side as with no columns to extract from v.
	 * 
	 * @param op The operation to this dictionary
	 * @param v  The values to apply on the dictionary (same number of cols as the dictionary)
	 * @return A new dictionary containing the updated values.
	 */
	public abstract ADictionary binOpRight(BinaryOperator op, double[] v);

	/**
	 * Apply the binary operator such that each value is offset by the reference before application. Then put the result
	 * into the new dictionary, but offset it by the new reference.
	 * 
	 * outValues[j] = op(this.values[j] + reference[i], v[colIndexes[i]]) - newReference[i]
	 * 
	 * @param op           The operation to apply on the dictionary values.
	 * @param v            The values to use on the right side of the operator.
	 * @param colIndexes   The column indexes to use.
	 * @param reference    The reference value to add before operator.
	 * @param newReference The reference value to subtract after operator.
	 * @return A new dictionary.
	 */
	public abstract ADictionary binOpRightWithReference(BinaryOperator op, double[] v, int[] colIndexes,
		double[] reference, double[] newReference);

	/**
	 * Returns a deep clone of the dictionary.
	 */
	public abstract ADictionary clone();

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
	 * 
	 * @param nrColumns The number of columns in the ColGroup to know how to get the values from the dictionary.
	 * @return a double array containing the row sums from this dictionary.
	 */
	public abstract double[] sumAllRowsToDouble(int nrColumns);

	/**
	 * Do exactly the same as the sumAllRowsToDouble but also sum the array given to a extra index in the end of the
	 * array.
	 * 
	 * @param defaultTuple The default row to sum in the end index returned.
	 * @return a double array containing the row sums from this dictionary.
	 */
	public abstract double[] sumAllRowsToDoubleWithDefault(double[] defaultTuple);

	/**
	 * Method used as a pre-aggregate of each tuple in the dictionary, to single double values with a reference.
	 * 
	 * @param reference The reference values to add to each cell.
	 * @return a double array containing the row sums from this dictionary.
	 */
	public abstract double[] sumAllRowsToDoubleWithReference(double[] reference);

	/**
	 * Method used as a pre-aggregate of each tuple in the dictionary, to single double values.
	 * 
	 * Note if the number of columns is one the actual dictionaries values are simply returned.
	 * 
	 * @param nrColumns The number of columns in the ColGroup to know how to get the values from the dictionary.
	 * @return a double array containing the row sums from this dictionary.
	 */
	public abstract double[] sumAllRowsToDoubleSq(int nrColumns);

	/**
	 * Method used as a pre-aggregate of each tuple in the dictionary, to single double values. But adds another cell to
	 * the return with an extra value that is the sum of the given defaultTuple.
	 * 
	 * @param defaultTuple The default row to sum in the end index returned.
	 * @return a double array containing the row sums from this dictionary.
	 */
	public abstract double[] sumAllRowsToDoubleSqWithDefault(double[] defaultTuple);

	/**
	 * Method used as a pre-aggregate of each tuple in the dictionary, to single double values.
	 * 
	 * @param reference The reference values to add to each cell.
	 * @return a double array containing the row sums from this dictionary.
	 */
	public abstract double[] sumAllRowsToDoubleSqWithReference(double[] reference);

	/**
	 * Method to product all rows to a column vector.
	 * 
	 * @param nrColumns The number of columns in the ColGroup to know how to get the values from the dictionary.
	 * @return A row product
	 */
	public abstract double[] productAllRowsToDouble(int nrColumns);

	/**
	 * Method to product all rows to a column vector with a default value added in the end.
	 * 
	 * @param defaultTuple The default row that aggregate to last cell
	 * @return A row product
	 */
	public abstract double[] productAllRowsToDoubleWithDefault(double[] defaultTuple);

	/**
	 * Method to product all rows to a column vector with a reference values added to all cells, and a reference product
	 * in the end
	 * 
	 * @param reference The reference row
	 * @return A row product
	 */
	public abstract double[] productAllRowsToDoubleWithReference(double[] reference);

	/**
	 * Get the column sum of the values contained in the dictionary
	 * 
	 * @param c          The output array allocated to contain all column groups output.
	 * @param counts     The counts of the individual tuples.
	 * @param colIndexes The columns indexes of the parent column group, this indicate where to put the column sum into
	 *                   the c output.
	 */
	public abstract void colSum(double[] c, int[] counts, int[] colIndexes);

	/**
	 * Get the column sum of the values contained in the dictionary
	 * 
	 * @param c          The output array allocated to contain all column groups output.
	 * @param counts     The counts of the individual tuples.
	 * @param colIndexes The columns indexes of the parent column group, this indicate where to put the column sum into
	 *                   the c output.
	 */
	public abstract void colSumSq(double[] c, int[] counts, int[] colIndexes);

	/**
	 * Get the column sum of the values contained in the dictionary with an offset reference value added to each cell.
	 * 
	 * @param c          The output array allocated to contain all column groups output.
	 * @param counts     The counts of the individual tuples.
	 * @param colIndexes The columns indexes of the parent column group, this indicate where to put the column sum into
	 *                   the c output.
	 * @param reference  The reference values to add to each cell.
	 */
	public abstract void colSumSqWithReference(double[] c, int[] counts, int[] colIndexes, double[] reference);

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
	public abstract double sumSq(int[] counts, int nCol);

	/**
	 * Get the square sum of the values contained in the dictionary with a reference offset on each value.
	 * 
	 * @param counts    The counts of the individual tuples
	 * @param reference The reference value
	 * @return The square sum scaled by the counts and reference.
	 */
	public abstract double sumSqWithReference(int[] counts, double[] reference);

	/**
	 * Get a string representation of the dictionary, that considers the layout of the data.
	 * 
	 * @param colIndexes The number of columns in the dictionary.
	 * @return A string that is nicer to print.
	 */
	public abstract String getString(int colIndexes);

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
	 * Detect if the dictionary contains a specific value.
	 * 
	 * @param pattern The value to search for
	 * @return true if the value is contained else false.
	 */
	public abstract boolean containsValue(double pattern);

	/**
	 * Detect if the dictionary contains a specific value with reference offset.
	 * 
	 * @param pattern   The pattern/ value to search for
	 * @param reference The reference double array.
	 * @return true if the value is contained else false.
	 */
	public abstract boolean containsValueWithReference(double pattern, double[] reference);

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
	 * Calculate the number of non zeros in the dictionary.
	 * 
	 * Each value in the dictionary should be added to the reference value.
	 * 
	 * The number of non zeros should be scaled with the given counts.
	 * 
	 * @param counts    The Counts of each dict entry.
	 * @param reference The reference vector.
	 * @param nRows     The number of rows in the input.
	 * @return The NonZero Count.
	 */
	public abstract long getNumberNonZerosWithReference(int[] counts, double[] reference, int nRows);

	/**
	 * Copies and adds the dictionary entry from this dictionary to the d dictionary
	 * 
	 * @param v    the target dictionary (dense double array)
	 * @param fr   the from index
	 * @param to   the to index
	 * @param nCol the number of columns
	 */
	public abstract void addToEntry(double[] v, int fr, int to, int nCol);

	/**
	 * copies and adds the dictonary entry from this dictionary yo the d dictionary rep times.
	 * 
	 * @param v    the target dictionary (dense double array)
	 * @param fr   the from index
	 * @param to   the to index
	 * @param nCol the number of columns
	 * @param rep  the number of repetitions to apply (simply multiply do not loop)
	 */
	public abstract void addToEntry(double[] v, int fr, int to, int nCol, int rep);

	public abstract void addToEntryVectorized(double[] v, int f1, int f2, int f3, int f4, int f5, int f6, int f7, int f8,
		int t1, int t2, int t3, int t4, int t5, int t6, int t7, int t8, int nCol);

	/**
	 * Allocate a new dictionary where the tuple given is subtracted from all tuples in the previous dictionary.
	 * 
	 * @param tuple a double list representing a tuple, it is given that the tuple with is the same as this dictionaries.
	 * @return a new instance of dictionary with the tuple subtracted.
	 */
	public abstract ADictionary subtractTuple(double[] tuple);

	/**
	 * Get this dictionary as a MatrixBlock dictionary. This allows us to use optimized kernels coded elsewhere in the
	 * system, such as matrix multiplication.
	 * 
	 * Return null if the matrix is empty.
	 * 
	 * @param nCol The number of columns contained in this column group.
	 * @return A Dictionary containing a MatrixBlock.
	 */
	public abstract MatrixBlockDictionary getMBDict(int nCol);

	/**
	 * Scale all tuples contained in the dictionary by the scaling factor given in the int list.
	 * 
	 * @param scaling The amount to multiply the given tuples with
	 * @param nCol    The number of columns contained in this column group.
	 * @return A New dictionary (since we don't want to modify the underlying dictionary)
	 */
	public abstract ADictionary scaleTuples(int[] scaling, int nCol);

	/**
	 * Pre Aggregate values for Right Matrix Multiplication.
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

	/**
	 * Make a copy of the values, and replace all values that match pattern with replacement value. If needed add a new
	 * column index. With reference such that each value in the dict is considered offset by the values contained in the
	 * reference.
	 * 
	 * @param pattern   The value to look for
	 * @param replace   The value to replace the other value with
	 * @param reference The reference tuple to add to all entries when replacing
	 * @return A new Column Group, reusing the index structure but with new values.
	 */
	public abstract ADictionary replaceWithReference(double pattern, double replace, double[] reference);

	/**
	 * Calculate the product of the dictionary weighted by counts.
	 * 
	 * @param ret    The result dense double array (containing one value)
	 * @param counts The count of individual tuples
	 * @param nCol   Number of columns in the dictionary.
	 */
	public abstract void product(double[] ret, int[] counts, int nCol);

	/**
	 * Calculate the product of the dictionary weighted by counts with a default value added .
	 * 
	 * @param ret      The result dense double array (containing one value)
	 * @param counts   The count of individual tuples
	 * @param def      The default tuple
	 * @param defCount The count of the default tuple
	 */
	public abstract void productWithDefault(double[] ret, int[] counts, double[] def, int defCount);

	/**
	 * Calculate the product of the dictionary weighted by counts and offset by reference
	 * 
	 * @param ret       The result dense double array (containing one value)
	 * @param counts    The counts of each entry in the dictionary
	 * @param reference The reference value.
	 * @param refCount  The number of occurrences of the ref value.
	 */
	public abstract void productWithReference(double[] ret, int[] counts, double[] reference, int refCount);

	/**
	 * Calculate the column product of the dictionary weighted by counts.
	 * 
	 * @param res        The result vector to put the result into
	 * @param counts     The weighted count of individual tuples
	 * @param colIndexes The column indexes.
	 */
	public abstract void colProduct(double[] res, int[] counts, int[] colIndexes);

	/**
	 * Calculate the column product of the dictionary weighted by counts.
	 * 
	 * @param res        The result vector to put the result into
	 * @param counts     The weighted count of individual tuples
	 * @param colIndexes The column indexes.
	 * @param reference  The reference value.
	 */
	public abstract void colProductWithReference(double[] res, int[] counts, int[] colIndexes, double[] reference);

	/**
	 * Central moment function to calculate the central moment of this column group. MUST be on a single column
	 * dictionary.
	 * 
	 * @param fn     The value function to apply
	 * @param counts The weight of individual tuples
	 * @param nRows  The number of rows in total of the column group
	 * @return The central moment Object
	 */
	public CM_COV_Object centralMoment(ValueFunction fn, int[] counts, int nRows) {
		return centralMoment(new CM_COV_Object(), fn, counts, nRows);
	}

	/**
	 * Central moment function to calculate the central moment of this column group. MUST be on a single column
	 * dictionary.
	 * 
	 * @param ret    The Central Moment object to be modified and returned
	 * @param fn     The value function to apply
	 * @param counts The weight of individual tuples
	 * @param nRows  The number of rows in total of the column group
	 * @return The central moment Object
	 */
	public abstract CM_COV_Object centralMoment(CM_COV_Object ret, ValueFunction fn, int[] counts, int nRows);

	/**
	 * Central moment function to calculate the central moment of this column group with a default offset on all missing
	 * tuples. MUST be on a single column dictionary.
	 * 
	 * @param fn     The value function to apply
	 * @param counts The weight of individual tuples
	 * @param def    The default values to offset the tuples with
	 * @param nRows  The number of rows in total of the column group
	 * @return The central moment Object
	 */
	public CM_COV_Object centralMomentWithDefault(ValueFunction fn, int[] counts, double def, int nRows) {
		return centralMomentWithDefault(new CM_COV_Object(), fn, counts, def, nRows);
	}

	/**
	 * Central moment function to calculate the central moment of this column group with a default offset on all missing
	 * tuples. MUST be on a single column dictionary.
	 * 
	 * @param ret    The Central Moment object to be modified and returned
	 * @param fn     The value function to apply
	 * @param counts The weight of individual tuples
	 * @param def    The default values to offset the tuples with
	 * @param nRows  The number of rows in total of the column group
	 * @return The central moment Object
	 */
	public abstract CM_COV_Object centralMomentWithDefault(CM_COV_Object ret, ValueFunction fn, int[] counts, double def,
		int nRows);

	/**
	 * Central moment function to calculate the central moment of this column group with a reference offset on each
	 * tuple. MUST be on a single column dictionary.
	 * 
	 * @param fn        The value function to apply
	 * @param counts    The weight of individual tuples
	 * @param reference The reference values to offset the tuples with
	 * @param nRows     The number of rows in total of the column group
	 * @return The central moment Object
	 */
	public CM_COV_Object centralMomentWithReference(ValueFunction fn, int[] counts, double reference, int nRows) {
		return centralMomentWithReference(new CM_COV_Object(), fn, counts, reference, nRows);
	}

	/**
	 * Central moment function to calculate the central moment of this column group with a reference offset on each
	 * tuple. MUST be on a single column dictionary.
	 * 
	 * @param ret       The Central Moment object to be modified and returned
	 * @param fn        The value function to apply
	 * @param counts    The weight of individual tuples
	 * @param reference The reference values to offset the tuples with
	 * @param nRows     The number of rows in total of the column group
	 * @return The central moment Object
	 */
	public abstract CM_COV_Object centralMomentWithReference(CM_COV_Object ret, ValueFunction fn, int[] counts,
		double reference, int nRows);

	/**
	 * Rexpand the dictionary (one hot encode)
	 * 
	 * @param max    the tuple width of the output
	 * @param ignore If we should ignore zero and negative values
	 * @param cast   If we should cast all double values to whole integer values
	 * @param nCol   The number of columns in the dictionary already (should be 1)
	 * @return A new dictionary
	 */
	public abstract ADictionary rexpandCols(int max, boolean ignore, boolean cast, int nCol);

	/**
	 * Rexpand the dictionary (one hot encode)
	 * 
	 * @param max       the tuple width of the output
	 * @param ignore    If we should ignore zero and negative values
	 * @param cast      If we should cast all double values to whole integer values
	 * @param reference A reference value to add to all tuples before expanding
	 * @return A new dictionary
	 */
	public abstract ADictionary rexpandColsWithReference(int max, boolean ignore, boolean cast, int reference);

	/**
	 * Get the sparsity of the dictionary.
	 * 
	 * @return a sparsity between 0 and 1
	 */
	public abstract double getSparsity();

	/**
	 * Multiply the v value with the dictionary entry at dictIdx and add it to the ret matrix at the columns specified in
	 * the int array.
	 * 
	 * @param v       Value to multiply
	 * @param ret     Output dense double array location
	 * @param off     Offset into the ret array that the "row" output starts at
	 * @param dictIdx The dictionary entry to multiply.
	 * @param cols    The columns to multiply into of the output.
	 */
	public abstract void multiplyScalar(double v, double[] ret, int off, int dictIdx, int[] cols);

	/**
	 * Transpose self matrix multiplication with a scaling factor on each pair of values.
	 * 
	 * @param counts The scaling factor
	 * @param rows   The row indexes
	 * @param cols   The col indexes
	 * @param ret    The output matrix block
	 */
	protected abstract void TSMMWithScaling(int[] counts, int[] rows, int[] cols, MatrixBlock ret);

	/**
	 * Matrix multiplication of dictionaries
	 * 
	 * Note the left is this, and it is transposed
	 * 
	 * @param right     Right hand side of multiplication
	 * @param rowsLeft  Offset rows on the left
	 * @param colsRight Offset cols on the right
	 * @param result    The output matrix block
	 */
	protected abstract void MMDict(ADictionary right, int[] rowsLeft, int[] colsRight, MatrixBlock result);

	/**
	 * Matrix multiplication of dictionaries left side dense and transposed right side is this.
	 * 
	 * @param left      Dense left side
	 * @param rowsLeft  Offset rows on the left
	 * @param colsRight Offset cols on the right
	 * @param result    The output matrix block
	 */
	protected abstract void MMDictDense(double[] left, int[] rowsLeft, int[] colsRight, MatrixBlock result);

	/**
	 * Matrix multiplication of dictionaries left side sparse and transposed right side is this.
	 * 
	 * @param left      Sparse left side
	 * @param rowsLeft  Offset rows on the left
	 * @param colsRight Offset cols on the right
	 * @param result    The output matrix block
	 */
	protected abstract void MMDictSparse(SparseBlock left, int[] rowsLeft, int[] colsRight, MatrixBlock result);

	/**
	 * Matrix multiplication but allocate output in upper triangle and twice if on diagonal, note this is left
	 * 
	 * @param right     Right side
	 * @param rowsLeft  Offset rows on the left
	 * @param colsRight Offset cols on the right
	 * @param result    The output matrix block
	 */
	protected abstract void TSMMToUpperTriangle(ADictionary right, int[] rowsLeft, int[] colsRight, MatrixBlock result);

	/**
	 * Matrix multiplication but allocate output in upper triangle and twice if on diagonal, note this is right
	 * 
	 * @param left      Dense left side
	 * @param rowsLeft  Offset rows on the left
	 * @param colsRight Offset cols on the right
	 * @param result    The output matrix block
	 */
	protected abstract void TSMMToUpperTriangleDense(double[] left, int[] rowsLeft, int[] colsRight, MatrixBlock result);

	/**
	 * Matrix multiplication but allocate output in upper triangle and twice if on diagonal, note this is right
	 * 
	 * @param left      Sparse left side
	 * @param rowsLeft  Offset rows on the left
	 * @param colsRight Offset cols on the right
	 * @param result    The output matrix block
	 */
	protected abstract void TSMMToUpperTriangleSparse(SparseBlock left, int[] rowsLeft, int[] colsRight,
		MatrixBlock result);

	/**
	 * Matrix multiplication but allocate output in upper triangle and twice if on diagonal, note this is left
	 * 
	 * @param right     Right side
	 * @param rowsLeft  Offset rows on the left
	 * @param colsRight Offset cols on the right
	 * @param scale     Scale factor
	 * @param result    The output matrix block
	 */
	protected abstract void TSMMToUpperTriangleScaling(ADictionary right, int[] rowsLeft, int[] colsRight, int[] scale,
		MatrixBlock result);

	/**
	 * Matrix multiplication but allocate output in upper triangle and twice if on diagonal, note this is right
	 * 
	 * @param left      Dense left side
	 * @param rowsLeft  Offset rows on the left
	 * @param colsRight Offset cols on the right
	 * @param scale     Scale factor
	 * @param result    The output matrix block
	 */
	protected abstract void TSMMToUpperTriangleDenseScaling(double[] left, int[] rowsLeft, int[] colsRight, int[] scale,
		MatrixBlock result);

	/**
	 * Matrix multiplication but allocate output in upper triangle and twice if on diagonal, note this is right
	 * 
	 * @param left      Sparse left side
	 * @param rowsLeft  Offset rows on the left
	 * @param colsRight Offset cols on the right
	 * @param scale     Scale factor
	 * @param result    The output matrix block
	 */
	protected abstract void TSMMToUpperTriangleSparseScaling(SparseBlock left, int[] rowsLeft, int[] colsRight,
		int[] scale, MatrixBlock result);

	protected String doubleToString(double v) {
		if(v == (long) v)
			return Long.toString(((long) v));
		else
			return Double.toString(v);
	}

	protected static void correctNan(double[] res, int[] colIndexes) {
		// since there is no nan values every in a dictionary, we exploit that
		// nan oly occur if we multiplied infinity with 0.
		for(int j = 0; j < colIndexes.length; j++)
			res[colIndexes[j]] = Double.isNaN(res[colIndexes[j]]) ? 0 : res[colIndexes[j]];
	}
}
