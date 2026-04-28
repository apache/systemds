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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.MathContext;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.utils.MemoryEstimates;

/**
 * This dictionary class aims to encapsulate the storage and operations over unique floating point values of a column
 * group. The primary reason for its introduction was to provide an entry point for specialization such as shared
 * dictionaries, which require additional information.
 */
public class Dictionary extends ACachingMBDictionary {

	private static final long serialVersionUID = -6517136537249507753L;

	protected final double[] _values;

	protected Dictionary(double[] values) {
		_values = values;
	}

	public static Dictionary create(double[] values) {
		if(values == null)
			throw new DMLCompressionException("Invalid construction of dictionary with null array");
		else if(values.length == 0)
			throw new DMLCompressionException("Invalid construction of dictionary with empty array");
		boolean nonZero = false;
		for(double d : values) {
			if(d != 0) {
				nonZero = true;
				break;
			}
		}
		return nonZero ? new Dictionary(values) : null;
	}

	public static Dictionary createNoCheck(double[] values) {
		return new Dictionary(values);
	}

	@Override
	public double[] getValues() {
		return _values;
	}

	@Override
	public double getValue(int i) {
		return _values[i];
	}

	@Override
	public final double getValue(int r, int c, int nCol) {
		return _values[r * nCol + c];
	}

	@Override
	public long getInMemorySize() {
		// object + values array + double
		return getInMemorySize(size());
	}

	public static long getInMemorySize(int valuesCount) {
		// object + values array
		return 16 + (long) MemoryEstimates.doubleArrayCost(valuesCount);
	}

	@Override
	public double aggregate(double init, Builtin fn) {
		// full aggregate can disregard tuple boundaries
		double ret = init;
		for(int i = 0; i < _values.length; i++)
			ret = fn.execute(ret, _values[i]);
		return ret;
	}

	@Override
	public double aggregateWithReference(double init, Builtin fn, double[] reference, boolean def) {
		final int nCol = reference.length;
		double ret = init;
		for(int i = 0; i < _values.length; i++)
			ret = fn.execute(ret, _values[i] + reference[i % nCol]);

		if(def)
			for(int i = 0; i < nCol; i++)
				ret = fn.execute(ret, reference[i]);

		return ret;
	}

	@Override
	public double[] aggregateRows(Builtin fn, int nCol) {
		if(nCol == 1)
			return _values;
		final int nRows = _values.length / nCol;
		double[] res = new double[nRows];
		for(int i = 0; i < nRows; i++) {
			final int off = i * nCol;
			res[i] = _values[off];
			for(int j = off + 1; j < off + nCol; j++)
				res[i] = fn.execute(res[i], _values[j]);
		}
		return res;
	}

	@Override
	public double[] aggregateRowsWithDefault(Builtin fn, double[] defaultTuple) {
		final int nCol = defaultTuple.length;
		final int nRows = _values.length / nCol;
		double[] res = new double[nRows + 1];
		for(int i = 0; i < nRows; i++) {
			final int off = i * nCol;
			res[i] = _values[off];
			for(int j = off + 1; j < off + nCol; j++)
				res[i] = fn.execute(res[i], _values[j]);
		}
		final int def = res.length - 1;
		res[def] = defaultTuple[0];
		for(int i = 1; i < nCol; i++)
			res[def] = fn.execute(res[def], defaultTuple[i]);

		return res;
	}

	@Override
	public double[] aggregateRowsWithReference(Builtin fn, double[] reference) {
		final int nCol = reference.length;
		final int nRows = _values.length / nCol;
		double[] res = new double[nRows + 1];
		int off = 0;
		for(int i = 0; i < nRows; i++) {
			res[i] = _values[off++] + reference[0];
			for(int j = 1; j < nCol; j++)
				res[i] = fn.execute(res[i], _values[off++] + reference[j]);
		}
		res[nRows] = reference[0];
		for(int i = 0; i < nCol; i++)
			res[nRows] = fn.execute(res[nRows], reference[i]);
		return res;
	}

	@Override
	public IDictionary applyScalarOp(ScalarOperator op) {
		if(op.fn instanceof Multiply)
			return applyScalarMultOp(op.getConstant());
		else
			return applyScalarGeneric(op);
	}

	private IDictionary applyScalarGeneric(ScalarOperator op) {
		final double[] retV = new double[_values.length];
		for(int i = 0; i < _values.length; i++)
			retV[i] = op.executeScalar(_values[i]);
		return create(retV);
	}

	private IDictionary applyScalarMultOp(double v) {
		final double[] retV = new double[_values.length];
		LibMatrixMult.vectMultiplyAdd(v, _values, retV, 0, 0, _values.length);
		return create(retV);
	}

	@Override
	public IDictionary applyScalarOpAndAppend(ScalarOperator op, double v0, int nCol) {
		final double[] retV = new double[_values.length + nCol];
		for(int i = 0; i < _values.length; i++)
			retV[i] = op.executeScalar(_values[i]);
		for(int i = _values.length; i < retV.length; i++)
			retV[i] = v0;
		return create(retV);
	}

	@Override
	public Dictionary applyUnaryOp(UnaryOperator op) {
		final double[] retV = new double[_values.length];
		for(int i = 0; i < _values.length; i++)
			retV[i] = op.fn.execute(_values[i]);
		return create(retV);
	}

	@Override
	public IDictionary applyUnaryOpAndAppend(UnaryOperator op, double v0, int nCol) {
		final double[] retV = new double[_values.length + nCol];
		for(int i = 0; i < _values.length; i++)
			retV[i] = op.fn.execute(_values[i]);
		for(int i = _values.length; i < retV.length; i++)
			retV[i] = v0;
		return create(retV);
	}

	@Override
	public Dictionary applyScalarOpWithReference(ScalarOperator op, double[] reference, double[] newReference) {
		final double[] retV = new double[_values.length];
		final int nCol = reference.length;
		final int nRow = _values.length / nCol;
		int off = 0;
		for(int i = 0; i < nRow; i++) {
			for(int j = 0; j < nCol; j++) {
				retV[off] = op.executeScalar(_values[off] + reference[j]) - newReference[j];
				off++;
			}
		}
		return create(retV);
	}

	@Override
	public IDictionary applyUnaryOpWithReference(UnaryOperator op, double[] reference, double[] newReference) {
		final double[] retV = new double[_values.length];
		final int nCol = reference.length;
		final int nRow = _values.length / nCol;
		int off = 0;
		for(int i = 0; i < nRow; i++) {
			for(int j = 0; j < nCol; j++) {
				retV[off] = op.fn.execute(_values[off] + reference[j]) - newReference[j];
				off++;
			}
		}
		return create(retV);
	}

	@Override
	public Dictionary binOpRight(BinaryOperator op, double[] v, IColIndex colIndexes) {
		final ValueFunction fn = op.fn;
		final double[] retVals = new double[_values.length];
		final int len = size();
		final int lenV = colIndexes.size();
		for(int i = 0; i < len; i++)
			retVals[i] = fn.execute(_values[i], v[colIndexes.get(i % lenV)]);

		return create(retVals);
	}

	@Override
	public Dictionary binOpRight(BinaryOperator op, double[] v) {
		final ValueFunction fn = op.fn;
		final double[] retVals = new double[_values.length];
		final int len = size();
		final int lenV = v.length;
		for(int i = 0; i < len; i++)
			retVals[i] = fn.execute(_values[i], v[i % lenV]);
		return create(retVals);
	}

	@Override
	public IDictionary binOpRightAndAppend(BinaryOperator op, double[] v, IColIndex colIndexes) {
		final ValueFunction fn = op.fn;
		final double[] retVals = new double[_values.length + colIndexes.size()];
		final int lenV = colIndexes.size();
		for(int i = 0; i < _values.length; i++)
			retVals[i] = fn.execute(_values[i], v[colIndexes.get(i % lenV)]);
		for(int i = _values.length; i < retVals.length; i++)
			retVals[i] = fn.execute(0, v[colIndexes.get(i % lenV)]);

		return create(retVals);
	}

	@Override
	public Dictionary binOpRightWithReference(BinaryOperator op, double[] v, IColIndex colIndexes, double[] reference,
		double[] newReference) {
		final ValueFunction fn = op.fn;
		final double[] retV = new double[_values.length];
		final int nCol = reference.length;
		final int nRow = _values.length / nCol;
		int off = 0;
		for(int i = 0; i < nRow; i++) {
			for(int j = 0; j < nCol; j++) {
				retV[off] = fn.execute(_values[off] + reference[j], v[colIndexes.get(j)]) - newReference[j];
				off++;
			}
		}
		return create(retV);
	}

	@Override
	public final Dictionary binOpLeft(BinaryOperator op, double[] v, IColIndex colIndexes) {
		final ValueFunction fn = op.fn;
		final double[] retVals = new double[_values.length];
		final int lenV = colIndexes.size();
		for(int i = 0; i < _values.length; i++)
			retVals[i] = fn.execute(v[colIndexes.get(i % lenV)], _values[i]);
		return create(retVals);
	}

	@Override
	public IDictionary binOpLeftAndAppend(BinaryOperator op, double[] v, IColIndex colIndexes) {
		final ValueFunction fn = op.fn;
		final double[] retVals = new double[_values.length + colIndexes.size()];
		final int lenV = colIndexes.size();
		for(int i = 0; i < _values.length; i++)
			retVals[i] = fn.execute(v[colIndexes.get(i % lenV)], _values[i]);
		for(int i = _values.length; i < retVals.length; i++)
			retVals[i] = fn.execute(v[colIndexes.get(i % lenV)], 0);

		return create(retVals);
	}

	@Override
	public Dictionary binOpLeftWithReference(BinaryOperator op, double[] v, IColIndex colIndexes, double[] reference,
		double[] newReference) {
		final ValueFunction fn = op.fn;
		final double[] retV = new double[_values.length];
		final int nCol = reference.length;
		final int nRow = _values.length / nCol;
		int off = 0;
		for(int i = 0; i < nRow; i++) {
			for(int j = 0; j < nCol; j++) {
				retV[off] = fn.execute(v[colIndexes.get(j)], _values[off] + reference[j]) - newReference[j];
				off++;
			}
		}
		return create(retV);
	}

	@Override
	public Dictionary clone() {
		return createNoCheck(_values.clone());
	}

	public static Dictionary read(DataInput in) throws IOException {
		int numVals = in.readInt();
		// read distinct values
		double[] values = new double[numVals];
		for(int i = 0; i < numVals; i++)
			values[i] = in.readDouble();
		return Dictionary.createNoCheck(values);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(DictionaryFactory.Type.FP64_DICT.ordinal());
		out.writeInt(size());
		for(int i = 0; i < size(); i++)
			out.writeDouble(_values[i]);
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + 4 + 8 * size();
	}

	private int size() {
		return _values.length;
	}

	@Override
	public int getNumberOfValues(int nCol) {
		return _values.length / nCol;
	}

	@Override
	public int getNumberOfColumns(int nrow) {
		return _values.length / nrow;
	}

	@Override
	public double[] sumAllRowsToDouble(int nrColumns) {
		if(nrColumns == 1)
			return getValues(); // shallow copy of values

		// pre-aggregate value tuple
		final int numVals = getNumberOfValues(nrColumns);
		double[] ret = new double[numVals];
		for(int k = 0; k < numVals; k++)
			ret[k] = sumRow(k, nrColumns);

		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleWithDefault(double[] defaultTuple) {
		final int nCol = defaultTuple.length;
		final int numVals = getNumberOfValues(nCol);
		final double[] ret = new double[numVals + 1];
		for(int k = 0; k < numVals; k++)
			ret[k] = sumRow(k, nCol);
		for(int i = 0; i < nCol; i++)
			ret[ret.length - 1] += defaultTuple[i];
		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleWithReference(double[] reference) {
		final int nCol = reference.length;
		final int numVals = getNumberOfValues(nCol);
		double[] ret = new double[numVals + 1];
		for(int k = 0; k < numVals; k++)
			ret[k] = sumRowWithReference(k, nCol, reference);
		for(int i = 0; i < nCol; i++)
			ret[numVals] += reference[i];
		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleSq(int nrColumns) {
		// pre-aggregate value tuple
		final int numVals = getNumberOfValues(nrColumns);
		final double[] ret = new double[numVals];
		for(int k = 0; k < numVals; k++)
			ret[k] = sumRowSq(k, nrColumns);

		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleSqWithDefault(double[] defaultTuple) {
		final int nCol = defaultTuple.length;
		final int numVals = getNumberOfValues(nCol);
		final double[] ret = new double[numVals + 1];
		for(int k = 0; k < numVals; k++)
			ret[k] = sumRowSq(k, nCol);
		for(int i = 0; i < nCol; i++)
			ret[ret.length - 1] += defaultTuple[i] * defaultTuple[i];
		return ret;
	}

	@Override
	public double[] productAllRowsToDouble(int nCol) {
		final int numVals = getNumberOfValues(nCol);
		final double[] ret = new double[numVals];
		for(int k = 0; k < numVals; k++)
			ret[k] = prodRow(k, nCol);
		return ret;
	}

	@Override
	public double[] productAllRowsToDoubleWithDefault(double[] defaultTuple) {
		final int nCol = defaultTuple.length;
		final int numVals = getNumberOfValues(nCol);
		final double[] ret = new double[numVals + 1];
		for(int k = 0; k < numVals; k++)
			ret[k] = prodRow(k, nCol);
		ret[numVals] = defaultTuple[0];
		for(int i = 1; i < nCol; i++)
			ret[numVals] *= defaultTuple[i];
		return ret;
	}

	@Override
	public double[] productAllRowsToDoubleWithReference(double[] reference) {
		final int nCol = reference.length;
		final int numVals = getNumberOfValues(nCol);
		final double[] ret = new double[numVals + 1];
		for(int k = 0; k < numVals; k++)
			ret[k] = prodRowWithReference(k, nCol, reference);
		ret[ret.length - 1] = reference[0];
		for(int i = 1; i < nCol; i++)
			ret[ret.length - 1] *= reference[i];
		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleSqWithReference(double[] reference) {
		final int nCol = reference.length;
		final int numVals = getNumberOfValues(nCol);
		double[] ret = new double[numVals + 1];
		for(int k = 0; k < numVals; k++)
			ret[k] = sumRowSqWithReference(k, nCol, reference);
		for(int i = 0; i < nCol; i++)
			ret[numVals] += reference[i] * reference[i];
		return ret;
	}

	private double sumRow(int k, int nrColumns) {
		final int valOff = k * nrColumns;
		double res = 0.0;
		for(int i = 0; i < nrColumns; i++)
			res += _values[valOff + i];
		return res;
	}

	public double sumRowWithReference(int k, int nrColumns, double[] reference) {
		final int valOff = k * nrColumns;
		double res = 0.0;
		for(int i = 0; i < nrColumns; i++)
			res += _values[valOff + i] + reference[i];
		return res;
	}

	private double sumRowSq(int k, int nrColumns) {
		final int valOff = k * nrColumns;
		double res = 0.0;
		for(int i = 0; i < nrColumns; i++)
			res += _values[valOff + i] * _values[valOff + i];
		return res;
	}

	private double prodRow(int k, int nrColumns) {
		final int valOff = k * nrColumns;
		final int end = valOff + nrColumns;
		double res = _values[valOff];
		for(int i = valOff + 1; i < end && res != 0; i++) // early abort on zero
			res *= _values[i];
		return res;
	}

	private double prodRowWithReference(int k, int nrColumns, double[] reference) {
		final int valOff = k * nrColumns;
		double res = _values[valOff] + reference[0];
		for(int i = 1; i < nrColumns; i++)
			res *= _values[valOff + i] + reference[i];
		return res;

	}

	private double sumRowSqWithReference(int k, int nrColumns, double[] reference) {
		final int valOff = k * nrColumns;
		double res = 0.0;
		for(int i = 0; i < nrColumns; i++) {
			final double v = _values[valOff + i] + reference[i];
			res += v * v;
		}
		return res;
	}

	@Override
	public void colSum(double[] c, int[] counts, IColIndex colIndexes) {
		final int nCol = colIndexes.size();
		for(int k = 0; k < counts.length; k++) {
			final int cntk = counts[k];
			final int off = k * nCol;
			for(int j = 0; j < nCol; j++)
				c[colIndexes.get(j)] += _values[off + j] * cntk;
		}
	}

	@Override
	public void colSumSq(double[] c, int[] counts, IColIndex colIndexes) {
		final int nCol = colIndexes.size();
		final int nRow = counts.length;
		int off = 0;
		for(int k = 0; k < nRow; k++) {
			final int cntk = counts[k];
			for(int j = 0; j < nCol; j++) {
				final double v = _values[off++];
				c[colIndexes.get(j)] += v * v * cntk;
			}
		}
	}

	@Override
	public void colProduct(double[] res, int[] counts, IColIndex colIndexes) {
		final int nCol = colIndexes.size();
		for(int k = 0; k < counts.length; k++) {
			final int cntk = counts[k];
			final int off = k * nCol;
			for(int j = 0; j < nCol; j++)
				res[colIndexes.get(j)] *= Math.pow(_values[off + j], cntk);
		}
		correctNan(res, colIndexes);
	}

	@Override
	public void colProductWithReference(double[] res, int[] counts, IColIndex colIndexes, double[] reference) {
		final int nCol = colIndexes.size();
		for(int k = 0; k < counts.length; k++) {
			final int cntk = counts[k];
			final int off = k * nCol;
			for(int j = 0; j < nCol; j++)
				res[colIndexes.get(j)] *= Math.pow(_values[off + j] + reference[j], cntk);
		}
		correctNan(res, colIndexes);
	}

	@Override
	public void colSumSqWithReference(double[] c, int[] counts, IColIndex colIndexes, double[] reference) {
		final int nCol = colIndexes.size();
		final int nRow = counts.length;
		int off = 0;
		for(int k = 0; k < nRow; k++) {
			final int cntk = counts[k];
			for(int j = 0; j < nCol; j++) {
				final double v = _values[off++] + reference[j];
				c[colIndexes.get(j)] += v * v * cntk;
			}
		}
	}

	@Override
	public double sum(int[] counts, int nCol) {
		double out = 0;
		int valOff = 0;
		for(int k = 0; k < counts.length; k++) {
			double rowSum = 0;
			int countK = counts[k];
			for(int j = 0; j < nCol; j++) {
				rowSum += _values[valOff++] * countK;
			}
			out += rowSum;
		}
		return out;
	}

	@Override
	public double sumSq(int[] counts, int nCol) {
		double out = 0;
		int valOff = 0;
		for(int k = 0; k < counts.length; k++) {
			final int countK = counts[k];
			for(int j = 0; j < nCol; j++) {
				final double val = _values[valOff++];
				out += val * val * countK;
			}
		}
		return out;
	}

	@Override
	public double sumSqWithReference(int[] counts, double[] reference) {
		final int nCol = reference.length;
		final int nRow = counts.length;
		double out = 0;
		int valOff = 0;
		for(int k = 0; k < nRow; k++) {
			final int countK = counts[k];
			for(int j = 0; j < nCol; j++) {
				final double val = _values[valOff++] + reference[j];
				out += val * val * countK;
			}
		}
		return out;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(_values.length * 3 + 10);
		sb.append("Dictionary : ");
		stringArray(sb, _values);
		return sb.toString();
	}

	private static void stringArray(StringBuilder sb, double[] val) {
		sb.append("[");
		if(val.length > 0) {
			sb.append(doubleToString(val[0]));
			for(int i = 1; i < val.length; i++) {
				sb.append(", ");
				sb.append(doubleToString(val[i]));
			}
		}
		sb.append("]");
	}

	public String getString(int colIndexes) {
		StringBuilder sb = new StringBuilder();
		if(colIndexes == 1)
			stringArray(sb, _values);
		else {
			sb.append("[\n\t");
			for(int i = 0; i < _values.length - 1; i++) {
				sb.append(doubleToString(_values[i]));
				sb.append((i) % (colIndexes) == colIndexes - 1 ? "\n\t" : ", ");
			}
			sb.append(doubleToString(_values[_values.length - 1]));
			sb.append("]");
		}
		return sb.toString();
	}

	public IDictionary sliceOutColumnRange(int idxStart, int idxEnd, int previousNumberOfColumns) {
		int numberTuples = getNumberOfValues(previousNumberOfColumns);
		int tupleLengthAfter = idxEnd - idxStart;
		double[] newDictValues = new double[tupleLengthAfter * numberTuples];
		int orgOffset = idxStart;
		int targetOffset = 0;
		for(int v = 0; v < numberTuples; v++) {
			for(int c = 0; c < tupleLengthAfter; c++, orgOffset++, targetOffset++) {
				newDictValues[targetOffset] = _values[orgOffset];
			}
			orgOffset += previousNumberOfColumns - idxEnd + idxStart;
		}
		return create(newDictValues);
	}

	@Override
	public boolean containsValue(double pattern) {
		boolean NaNpattern = Double.isNaN(pattern);

		if(NaNpattern) {
			for(double v : _values)
				if(Double.isNaN(v))
					return true;
		}
		else {
			for(double v : _values)
				if(v == pattern)
					return true;
		}

		return false;
	}

	@Override
	public boolean containsValueWithReference(double pattern, double[] reference) {
		if(Double.isNaN(pattern))
			return super.containsValueWithReference(pattern, reference);
		final int nCol = reference.length;
		for(int i = 0; i < _values.length; i++)
			if(_values[i] + reference[i % nCol] == pattern)
				return true;
		return false;
	}

	@Override
	public long getNumberNonZeros(int[] counts, int nCol) {
		long nnz = 0;
		final int nRow = counts.length;
		for(int i = 0; i < nRow; i++) {
			long rowCount = 0;
			final int off = i * nCol;
			for(int j = off; j < off + nCol; j++) {
				if(_values[j] != 0)
					rowCount++;
			}
			nnz += rowCount * counts[i];
		}
		return nnz;
	}

	@Override
	public int[] countNNZZeroColumns(int[] counts) {
		final int nRow = counts.length;
		final int nCol = _values.length / nRow;

		final int[] ret = new int[nCol];
		for(int i = 0; i < nRow; i++) {
			for(int j = 0; j < nCol; j++) {
				final int off = i * nCol + j;
				if(_values[off] != 0)
					ret[j] += counts[i];
			}
		}
		return ret;
	}

	@Override
	public long getNumberNonZerosWithReference(int[] counts, double[] reference, int nRows) {
		long nnz = 0;
		final int nCol = reference.length;
		final int nRow = counts.length;

		for(int i = 0; i < nRow; i++) {
			long rowCount = 0;
			final int off = i * nCol;
			for(int j = off, jj = 0; j < off + nCol; j++, jj++) {
				if(_values[j] + reference[jj] != 0)
					rowCount++;
			}
			nnz += rowCount * counts[i];
		}

		return nnz;
	}

	@Override
	public final void addToEntry(double[] v, int fr, int to, int nCol) {
		final int sf = fr * nCol; // start from
		final int st = to * nCol; // start to
		addToOffsets(v, sf, st, nCol);
	}

	private final void addToOffsets(final double[] v, final int sf, final int st, final int nCol) {
		for(int i = sf, j = st; i < sf + nCol; i++, j++)
			v[j] += _values[i];
	}

	@Override
	public final void addToEntry(double[] v, int fr, int to, int nCol, int rep) {
		final int sf = fr * nCol; // start from
		final int st = to * nCol; // start to
		addToOffsets(v, sf, st, nCol, rep);
	}

	private final void addToOffsets(double[] v, int sf, int st, int nCol, int rep) {
		for(int i = sf, j = st; i < sf + nCol; i++, j++)
			v[j] += _values[i] * rep;
	}

	@Override
	public void addToEntryVectorized(double[] v, int f1, int f2, int f3, int f4, int f5, int f6, int f7, int f8, int t1,
		int t2, int t3, int t4, int t5, int t6, int t7, int t8, int nCol) {
		addToOffsets(v, f1 * nCol, t1 * nCol, nCol);
		addToOffsets(v, f2 * nCol, t2 * nCol, nCol);
		addToOffsets(v, f3 * nCol, t3 * nCol, nCol);
		addToOffsets(v, f4 * nCol, t4 * nCol, nCol);
		addToOffsets(v, f5 * nCol, t5 * nCol, nCol);
		addToOffsets(v, f6 * nCol, t6 * nCol, nCol);
		addToOffsets(v, f7 * nCol, t7 * nCol, nCol);
		addToOffsets(v, f8 * nCol, t8 * nCol, nCol);
	}

	@Override
	public DictType getDictType() {
		return DictType.Dict;
	}

	@Override
	public IDictionary subtractTuple(double[] tuple) {
		double[] newValues = new double[_values.length];
		for(int i = 0; i < _values.length;)
			for(int j = 0; j < tuple.length; i++, j++)
				newValues[i] = _values[i] - tuple[j];

		return create(newValues);
	}

	@Override
	public MatrixBlockDictionary createMBDict(int nCol) {
		MatrixBlockDictionary ret = MatrixBlockDictionary.createDictionary(_values, nCol, true);
		return ret;
	}

	@Override
	public void aggregateCols(double[] c, Builtin fn, IColIndex colIndexes) {
		final int nCol = colIndexes.size();
		final int rlen = _values.length / nCol;
		for(int k = 0; k < rlen; k++)
			for(int j = 0, valOff = k * nCol; j < nCol; j++)
				c[colIndexes.get(j)] = fn.execute(c[colIndexes.get(j)], _values[valOff + j]);
	}

	@Override
	public void aggregateColsWithReference(double[] c, Builtin fn, IColIndex colIndexes, double[] reference,
		boolean def) {
		final int nCol = reference.length;
		final int rlen = _values.length / nCol;
		for(int k = 0; k < rlen; k++)
			for(int j = 0, valOff = k * nCol; j < nCol; j++)
				c[colIndexes.get(j)] = fn.execute(c[colIndexes.get(j)], _values[valOff + j] + reference[j]);
		if(def)
			for(int i = 0; i < nCol; i++) {
				final int cix = colIndexes.get(i);
				c[cix] = fn.execute(c[cix], reference[i]);
			}
	}

	@Override
	public IDictionary scaleTuples(int[] scaling, int nCol) {
		final double[] scaledValues = new double[_values.length];
		int off = 0;
		for(int tuple = 0; tuple < _values.length / nCol; tuple++) {
			final int scale = scaling[tuple];
			for(int v = 0; v < nCol; v++) {
				scaledValues[off] = _values[off] * scale;
				off++;
			}
		}
		return create(scaledValues);
	}

	@Override
	public Dictionary preaggValuesFromDense(int numVals, IColIndex colIndexes, IColIndex aggregateColumns, double[] b,
		int cut) {
		final int cz = colIndexes.size();
		final int az = aggregateColumns.size();
		final double[] ret = new double[numVals * az];
		for(int k = 0, off = 0; k < numVals * cz; k += cz, off += az) {
			for(int h = 0; h < cz; h++) {
				final int idb = colIndexes.get(h) * cut;
				double v = _values[k + h];
				if(v != 0)
					for(int i = 0; i < az; i++)
						ret[off + i] += v * b[idb + aggregateColumns.get(i)];
			}
		}
		return create(ret);
	}

	@Override
	public IDictionary replace(double pattern, double replace, int nCol) {
		double[] retV = new double[_values.length];
		for(int i = 0; i < _values.length; i++) {
			final double v = _values[i];
			retV[i] = Util.eq(v, pattern) ? replace : v;
		}
		return create(retV);
	}

	@Override
	public IDictionary replaceWithReference(double pattern, double replace, double[] reference) {
		final int nCol = reference.length;
		final int nRow = _values.length / nCol;
		if(Util.eq(pattern, Double.NaN)) {
			return replaceWithReferenceNaN(replace, reference, nCol, nRow);
		}
		else {
			final double[] retV = new double[_values.length];
			int off = 0;
			for(int i = 0; i < nRow; i++) {
				for(int j = 0; j < nCol; j++) {
					final double ref = reference[j];
					final double v = _values[off];
					retV[off] = Math.abs(v + ref - pattern) < 0.000001 ? replace - ref : v;
					off++;
				}
			}
			return create(retV);
		}
	}

	private IDictionary replaceWithReferenceNaN(double replace, double[] reference, final int nCol, final int nRow) {
		final Set<Integer> colsWithNan = getColsWithNan(replace, reference);
		final double[] retV;
		if(colsWithNan != null) {
			if(colsWithNan.size() == nCol && replace == 0)
				return null;
			retV = new double[_values.length];
			replaceWithReferenceNanDenseWithNanCols(replace, reference, nRow, nCol, colsWithNan, _values, retV);
		}
		else {
			retV = new double[_values.length];
			replaceWithReferenceNanDenseWithoutNanCols(replace, reference, nRow, nCol, retV, _values);
		}
		return create(retV);
	}

	protected static Set<Integer> getColsWithNan(double replace, double[] reference) {
		Set<Integer> colsWithNan = null;
		for(int i = 0; i < reference.length; i++) {
			if(Util.eq(reference[i], Double.NaN)) {
				if(colsWithNan == null)
					colsWithNan = new HashSet<>();
				colsWithNan.add(i);
				reference[i] = replace;
			}
		}
		return colsWithNan;
	}

	protected static void replaceWithReferenceNanDenseWithoutNanCols(final double replace, final double[] reference,
		final int nRow, final int nCol, final double[] retV, final double[] values) {
		int off = 0;
		for(int i = 0; i < nRow; i++) {
			for(int j = 0; j < nCol; j++) {
				final double v = values[off];
				retV[off++] = Util.eq(Double.NaN, v) ? replace - reference[j] : v;
			}
		}
	}

	protected static void replaceWithReferenceNanDenseWithNanCols(final double replace, final double[] reference,
		final int nRow, final int nCol, Set<Integer> colsWithNan, final double[] values, final double[] retV) {
		int off = 0;
		for(int i = 0; i < nRow; i++) {
			for(int j = 0; j < nCol; j++) {
				final double v = values[off];
				if(colsWithNan.contains(j))
					retV[off++] = 0;
				else if(Util.eq(v, Double.NaN))
					retV[off++] = replace - reference[j];
				else
					retV[off++] = v;
			}
		}
	}

	@Override
	public void product(double[] ret, int[] counts, int nCol) {
		if(ret[0] == 0)
			return;
		final MathContext cont = MathContext.DECIMAL128;
		final int len = counts.length;
		BigDecimal tmp = BigDecimal.ONE;
		for(int i = 0; i < len; i++) {
			for(int j = i * nCol; j < (i + 1) * nCol; j++) {
				double v = _values[j];
				if(v == 0) {
					ret[0] = 0;
					return;
				}
				tmp = tmp.multiply(new BigDecimal(v).pow(counts[i], cont), cont);
			}
		}
		if(Math.abs(tmp.doubleValue()) == 0)
			ret[0] = 0;

		else if(!Double.isInfinite(ret[0]))
			ret[0] = new BigDecimal(ret[0]).multiply(tmp, MathContext.DECIMAL128).doubleValue();
	}

	@Override
	public void productWithDefault(double[] ret, int[] counts, double[] def, int defCount) {
		if(ret[0] == 0)
			return;
		final MathContext cont = MathContext.DECIMAL128;
		final int len = counts.length;
		final int nCol = def.length;
		BigDecimal tmp = BigDecimal.ONE;
		for(int i = 0; i < len; i++) {
			for(int j = i * nCol; j < (i + 1) * nCol; j++) {
				double v = _values[j];
				if(v == 0) {
					ret[0] = 0;
					return;
				}
				tmp = tmp.multiply(new BigDecimal(v).pow(counts[i], cont), cont);
			}
		}
		for(int x = 0; x < def.length; x++)
			tmp = tmp.multiply(new BigDecimal(def[x]).pow(defCount, cont), cont);
		if(Math.abs(tmp.doubleValue()) == 0)
			ret[0] = 0;
		else if(!Double.isInfinite(ret[0]))
			ret[0] = new BigDecimal(ret[0]).multiply(tmp, MathContext.DECIMAL128).doubleValue();
	}

	@Override
	public void productWithReference(double[] ret, int[] counts, double[] reference, int refCount) {
		if(ret[0] == 0)
			return;
		final MathContext cont = MathContext.DECIMAL128;
		final int nRow = counts.length;
		final int nCol = reference.length;

		BigDecimal tmp = BigDecimal.ONE;
		int off = 0;
		for(int i = 0; i < nRow; i++) {
			for(int j = 0; j < nCol; j++) {
				final double v = _values[off++] + reference[j];
				if(v == 0) {
					ret[0] = 0;
					return;
				}
				else if(!Double.isFinite(v)) {
					ret[0] = v;
					return;
				}
				tmp = tmp.multiply(new BigDecimal(v).pow(counts[i], cont), cont);
			}
		}
		for(int x = 0; x < reference.length; x++)
			tmp = tmp.multiply(new BigDecimal(reference[x]).pow(refCount, cont), cont);
		if(Math.abs(tmp.doubleValue()) == 0)
			ret[0] = 0;
		else if(!Double.isInfinite(ret[0]))
			ret[0] = new BigDecimal(ret[0]).multiply(tmp, MathContext.DECIMAL128).doubleValue();

	}

	@Override
	public CmCovObject centralMoment(CmCovObject ret, ValueFunction fn, int[] counts, int nRows) {
		// should be guaranteed to only contain one value per tuple in dictionary.
		for(int i = 0; i < _values.length; i++)
			fn.execute(ret, _values[i], counts[i]);

		if(ret.getWeight() < nRows)
			fn.execute(ret, 0, nRows - ret.getWeight());
		return ret;
	}

	@Override
	public CmCovObject centralMomentWithDefault(CmCovObject ret, ValueFunction fn, int[] counts, double def,
		int nRows) {
		// should be guaranteed to only contain one value per tuple in dictionary.
		for(int i = 0; i < _values.length; i++)
			fn.execute(ret, _values[i], counts[i]);

		if(ret.getWeight() < nRows)
			fn.execute(ret, def, nRows - ret.getWeight());
		return ret;
	}

	@Override
	public CmCovObject centralMomentWithReference(CmCovObject ret, ValueFunction fn, int[] counts, double reference,
		int nRows) {
		// should be guaranteed to only contain one value per tuple in dictionary.
		for(int i = 0; i < _values.length; i++)
			fn.execute(ret, _values[i] + reference, counts[i]);

		if(ret.getWeight() < nRows)
			fn.execute(ret, reference, nRows - ret.getWeight());
		return ret;
	}

	@Override
	public IDictionary rexpandCols(int max, boolean ignore, boolean cast, int nCol) {
		if(nCol > 1)
			throw new DMLCompressionException("Invalid to rexpand the column groups if more than one column");
		MatrixBlockDictionary m = getMBDict(nCol);
		return m == null ? null : m.rexpandCols(max, ignore, cast, nCol);
	}

	@Override
	public IDictionary rexpandColsWithReference(int max, boolean ignore, boolean cast, int reference) {
		MatrixBlockDictionary m = getMBDict(1);
		if(m == null)
			return null;
		IDictionary a = m.applyScalarOp(new RightScalarOperator(Plus.getPlusFnObject(), reference));
		if(a == null)
			return null; // second ending
		a = a.rexpandCols(max, ignore, cast, 1);
		return a;
	}

	@Override
	public double getSparsity() {
		int zeros = 0;
		for(double v : _values)
			if(v == 0.0)
				zeros++;
		return OptimizerUtils.getSparsity(_values.length, 1L, _values.length - zeros);
	}

	@Override
	public void multiplyScalar(double v, double[] ret, int off, int dictIdx, IColIndex cols) {
		final int offD = dictIdx * cols.size();
		for(int i = 0; i < cols.size(); i++) {
			double a = v * _values[offD + i];
			ret[off + cols.get(i)] += a;
		}
	}

	@Override
	public void TSMMWithScaling(int[] counts, IColIndex rows, IColIndex cols, MatrixBlock ret) {
		DictLibMatrixMult.TSMMDictsDenseWithScaling(_values, rows, cols, counts, ret);
	}

	@Override
	public void MMDict(IDictionary right, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		right.MMDictDense(_values, rowsLeft, colsRight, result);
	}

	@Override
	public void MMDictScaling(IDictionary right, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result,
		int[] scaling) {
		right.MMDictScalingDense(_values, rowsLeft, colsRight, result, scaling);
	}

	@Override
	public void MMDictDense(double[] left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		DictLibMatrixMult.MMDictsDenseDense(left, _values, rowsLeft, colsRight, result);
	}

	@Override
	public void MMDictScalingDense(double[] left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result,
		int[] scaling) {
		DictLibMatrixMult.MMDictsScalingDenseDense(left, _values, rowsLeft, colsRight, result, scaling);
	}

	@Override
	public void MMDictSparse(SparseBlock left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		DictLibMatrixMult.MMDictsSparseDense(left, _values, rowsLeft, colsRight, result);
	}

	@Override
	public void MMDictScalingSparse(SparseBlock left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result,
		int[] scaling) {
		DictLibMatrixMult.MMDictsScalingSparseDense(left, _values, rowsLeft, colsRight, result, scaling);
	}

	@Override
	public void TSMMToUpperTriangle(IDictionary right, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		right.TSMMToUpperTriangleDense(_values, rowsLeft, colsRight, result);
	}

	@Override
	public void TSMMToUpperTriangleDense(double[] left, IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		DictLibMatrixMult.MMToUpperTriangleDenseDense(left, _values, rowsLeft, colsRight, result);
	}

	@Override
	public void TSMMToUpperTriangleSparse(SparseBlock left, IColIndex rowsLeft, IColIndex colsRight,
		MatrixBlock result) {
		DictLibMatrixMult.MMToUpperTriangleSparseDense(left, _values, rowsLeft, colsRight, result);
	}

	@Override
	public void TSMMToUpperTriangleScaling(IDictionary right, IColIndex rowsLeft, IColIndex colsRight, int[] scale,
		MatrixBlock result) {
		right.TSMMToUpperTriangleDenseScaling(_values, rowsLeft, colsRight, scale, result);
	}

	@Override
	public void TSMMToUpperTriangleDenseScaling(double[] left, IColIndex rowsLeft, IColIndex colsRight, int[] scale,
		MatrixBlock result) {
		DictLibMatrixMult.TSMMToUpperTriangleDenseDenseScaling(left, _values, rowsLeft, colsRight, scale, result);
	}

	@Override
	public void TSMMToUpperTriangleSparseScaling(SparseBlock left, IColIndex rowsLeft, IColIndex colsRight, int[] scale,
		MatrixBlock result) {
		DictLibMatrixMult.TSMMToUpperTriangleSparseDenseScaling(left, _values, rowsLeft, colsRight, scale, result);
	}

	@Override
	public boolean equals(IDictionary o) {
		if(o instanceof Dictionary)
			return Arrays.equals(_values, ((Dictionary) o)._values);
		else if(o != null)
			return o.equals(this);
		return false;
	}

	@Override
	public IDictionary cbind(IDictionary that, int nCol) {
		int nRowThat = that.getNumberOfValues(nCol);
		int nColThis = _values.length / nRowThat;
		MatrixBlockDictionary mbd = getMBDict(nColThis);
		return mbd.cbind(that, nCol);
	}

	@Override
	public IDictionary reorder(int[] reorder) {
		double[] retV = new double[_values.length];
		Dictionary ret = new Dictionary(retV);
		int nRows = _values.length / reorder.length;

		for(int r = 0; r < nRows; r++) {
			int off = r * reorder.length;
			for(int c = 0; c < reorder.length; c++)
				retV[off + c] = _values[off + reorder[c]];
		}
		return ret;
	}

	@Override
	protected IDictionary rightMMPreAggSparseSelectedCols(int numVals, SparseBlock b, IColIndex thisCols,
		IColIndex aggregateColumns) {

		final int thisColsSize = thisCols.size();
		final int aggColSize = aggregateColumns.size();
		final double[] ret = new double[numVals * aggColSize];

		for(int h = 0; h < thisColsSize; h++) {
			// chose row in right side matrix via column index of the dictionary
			final int colIdx = thisCols.get(h);
			if(b.isEmpty(colIdx))
				continue;

			// extract the row values on the right side.
			final double[] sValues = b.values(colIdx);
			final int[] sIndexes = b.indexes(colIdx);
			final int sPos = b.pos(colIdx);
			final int sEnd = b.size(colIdx) + sPos;

			for(int j = 0; j < numVals; j++) { // rows left
				final int offOut = j * aggColSize;
				final double v = getValue(j, h, thisColsSize);
				sparseAddSelected(sPos, sEnd, aggColSize, aggregateColumns, sIndexes, sValues, ret, offOut, v);
			}

		}
		return Dictionary.create(ret);
	}

	private void sparseAddSelected(int sPos, int sEnd, int aggColSize, IColIndex aggregateColumns, int[] sIndexes,
		double[] sValues, double[] ret, int offOut, double v) {

		int retIdx = 0;
		for(int i = sPos; i < sEnd; i++) {
			// skip through the retIdx.
			while(retIdx < aggColSize && aggregateColumns.get(retIdx) < sIndexes[i])
				retIdx++;
			if(retIdx == aggColSize)
				break;
			ret[offOut + retIdx] += v * sValues[i];
		}
		retIdx = 0;
	}

	@Override
	protected IDictionary rightMMPreAggSparseAllColsRight(int numVals, SparseBlock b, IColIndex thisCols,
		int nColRight) {
		final int thisColsSize = thisCols.size();
		final double[] ret = new double[numVals * nColRight];

		for(int h = 0; h < thisColsSize; h++) { // common dim
			// chose row in right side matrix via column index of the dictionary
			final int colIdx = thisCols.get(h);
			if(b.isEmpty(colIdx))
				continue;

			// extract the row values on the right side.
			final double[] sValues = b.values(colIdx);
			final int[] sIndexes = b.indexes(colIdx);
			final int sPos = b.pos(colIdx);
			final int sEnd = b.size(colIdx) + sPos;

			for(int i = 0; i < numVals; i++) { // rows left
				final int offOut = i * nColRight;
				final double v = getValue(i, h, thisColsSize);
				SparseAdd(sPos, sEnd, ret, offOut, sIndexes, sValues, v);
			}
		}
		return Dictionary.create(ret);
	}

	private void SparseAdd(int sPos, int sEnd, double[] ret, int offOut, int[] sIdx, double[] sVals, double v) {
		if(v != 0) {
			for(int k = sPos; k < sEnd; k++) { // cols right with value
				ret[offOut + sIdx[k]] += v * sVals[k];
			}
		}
	}

	@Override
	public IDictionary append(double[] row) {
		double[] retV = new double[_values.length + row.length];
		System.arraycopy(_values, 0, retV, 0, _values.length);
		System.arraycopy(row, 0, retV, _values.length, row.length);
		return new Dictionary(retV);
	}

}
