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

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.utils.MemoryEstimates;

/**
 * This dictionary class aims to encapsulate the storage and operations over unique floating point values of a column
 * group. The primary reason for its introduction was to provide an entry point for specialization such as shared
 * dictionaries, which require additional information.
 */
public class Dictionary extends ADictionary {

	private static final long serialVersionUID = -6517136537249507753L;

	protected final double[] _values;

	protected Dictionary(double[] values) {
		if(values == null || values.length == 0)
			throw new DMLCompressionException("Invalid construction of dictionary with null array");
		_values = values;
	}

	public static Dictionary create(double[] values) {
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

	protected static long getInMemorySize(int valuesCount) {
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
	public Dictionary applyScalarOp(ScalarOperator op) {
		final double[] retV = new double[_values.length];
		for(int i = 0; i < _values.length; i++)
			retV[i] = op.executeScalar(_values[i]);
		return create(retV);
	}

	@Override
	public ADictionary applyScalarOpAndAppend(ScalarOperator op, double v0, int nCol) {
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
	public ADictionary applyUnaryOpAndAppend(UnaryOperator op, double v0, int nCol) {
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
	public ADictionary applyUnaryOpWithReference(UnaryOperator op, double[] reference, double[] newReference) {
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
	public Dictionary binOpRight(BinaryOperator op, double[] v, int[] colIndexes) {
		final ValueFunction fn = op.fn;
		final double[] retVals = new double[_values.length];
		final int len = size();
		final int lenV = colIndexes.length;
		for(int i = 0; i < len; i++)
			retVals[i] = fn.execute(_values[i], v[colIndexes[i % lenV]]);
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
	public ADictionary binOpRightAndAppend(BinaryOperator op, double[] v, int[] colIndexes) {
		final ValueFunction fn = op.fn;
		final double[] retVals = new double[_values.length + colIndexes.length];
		final int lenV = colIndexes.length;
		for(int i = 0; i < _values.length; i++)
			retVals[i] = fn.execute(_values[i], v[colIndexes[i % lenV]]);
		for(int i = _values.length; i < _values.length; i++)
			retVals[i] = fn.execute(0, v[colIndexes[i % lenV]]);

		return create(retVals);
	}

	@Override
	public Dictionary binOpRightWithReference(BinaryOperator op, double[] v, int[] colIndexes, double[] reference,
		double[] newReference) {
		final ValueFunction fn = op.fn;
		final double[] retV = new double[_values.length];
		final int nCol = reference.length;
		final int nRow = _values.length / nCol;
		int off = 0;
		for(int i = 0; i < nRow; i++) {
			for(int j = 0; j < nCol; j++) {
				retV[off] = fn.execute(_values[off] + reference[j], v[colIndexes[j]]) - newReference[j];
				off++;
			}
		}
		return create(retV);
	}

	@Override
	public final Dictionary binOpLeft(BinaryOperator op, double[] v, int[] colIndexes) {
		final ValueFunction fn = op.fn;
		final double[] retVals = new double[_values.length];
		final int lenV = colIndexes.length;
		for(int i = 0; i < _values.length; i++)
			retVals[i] = fn.execute(v[colIndexes[i % lenV]], _values[i]);
		return create(retVals);
	}

	@Override
	public ADictionary binOpLeftAndAppend(BinaryOperator op, double[] v, int[] colIndexes) {
		final ValueFunction fn = op.fn;
		final double[] retVals = new double[_values.length + colIndexes.length];
		final int lenV = colIndexes.length;
		for(int i = 0; i < _values.length; i++)
			retVals[i] = fn.execute(v[colIndexes[i % lenV]], _values[i]);
		for(int i = _values.length; i < _values.length; i++)
			retVals[i] = fn.execute(v[colIndexes[i % lenV]], 0);

		return create(retVals);
	}

	@Override
	public Dictionary binOpLeftWithReference(BinaryOperator op, double[] v, int[] colIndexes, double[] reference,
		double[] newReference) {
		final ValueFunction fn = op.fn;
		final double[] retV = new double[_values.length];
		final int nCol = reference.length;
		final int nRow = _values.length / nCol;
		int off = 0;
		for(int i = 0; i < nRow; i++) {
			for(int j = 0; j < nCol; j++) {
				retV[off] = fn.execute(v[colIndexes[j]], _values[off] + reference[j]) - newReference[j];
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
		ret[ret.length - 1] = defaultTuple[0];
		for(int i = 1; i < nCol; i++)
			ret[ret.length - 1] *= defaultTuple[i];
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
		double res = _values[valOff];
		for(int i = 1; i < nrColumns; i++)
			res *= _values[valOff + i];
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
	public void colSum(double[] c, int[] counts, int[] colIndexes) {
		final int nCol = colIndexes.length;
		for(int k = 0; k < counts.length; k++) {
			final int cntk = counts[k];
			final int off = k * nCol;
			for(int j = 0; j < nCol; j++)
				c[colIndexes[j]] += _values[off + j] * cntk;
		}
	}

	@Override
	public void colSumSq(double[] c, int[] counts, int[] colIndexes) {
		final int nCol = colIndexes.length;
		final int nRow = counts.length;
		int off = 0;
		for(int k = 0; k < nRow; k++) {
			final int cntk = counts[k];
			for(int j = 0; j < nCol; j++) {
				final double v = _values[off++];
				c[colIndexes[j]] += v * v * cntk;
			}
		}
	}

	@Override
	public void colProduct(double[] res, int[] counts, int[] colIndexes) {
		final int nCol = colIndexes.length;
		for(int k = 0; k < counts.length; k++) {
			final int cntk = counts[k];
			final int off = k * nCol;
			for(int j = 0; j < nCol; j++)
				res[colIndexes[j]] *= Math.pow(_values[off + j], cntk);
		}
		correctNan(res, colIndexes);
	}

	@Override
	public void colProductWithReference(double[] res, int[] counts, int[] colIndexes, double[] reference) {
		final int nCol = colIndexes.length;
		for(int k = 0; k < counts.length; k++) {
			final int cntk = counts[k];
			final int off = k * nCol;
			for(int j = 0; j < nCol; j++)
				res[colIndexes[j]] *= Math.pow(_values[off + j] + reference[j], cntk);
		}

		correctNan(res, colIndexes);
	}

	@Override
	public void colSumSqWithReference(double[] c, int[] counts, int[] colIndexes, double[] reference) {
		final int nCol = colIndexes.length;
		final int nRow = counts.length;
		int off = 0;
		for(int k = 0; k < nRow; k++) {
			final int cntk = counts[k];
			for(int j = 0; j < nCol; j++) {
				final double v = _values[off++] + reference[j];
				c[colIndexes[j]] += v * v * cntk;
			}
		}
	}

	@Override
	public double sum(int[] counts, int nCol) {
		double out = 0;
		int valOff = 0;
		for(int k = 0; k < counts.length; k++) {
			int countK = counts[k];
			for(int j = 0; j < nCol; j++) {
				out += _values[valOff++] * countK;
			}
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
		StringBuilder sb = new StringBuilder();

		sb.append("Dictionary : ");
		sb.append(Arrays.toString(_values));
		return sb.toString();
	}

	public String getString(int colIndexes) {
		StringBuilder sb = new StringBuilder();
		if(colIndexes == 1)
			sb.append(Arrays.toString(_values));
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

	public ADictionary sliceOutColumnRange(int idxStart, int idxEnd, int previousNumberOfColumns) {
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
	public boolean isLossy() {
		return false;
	}

	@Override
	public ADictionary subtractTuple(double[] tuple) {
		double[] newValues = new double[_values.length];
		for(int i = 0; i < _values.length;)
			for(int j = 0; j < tuple.length; i++, j++)
				newValues[i] = _values[i] - tuple[j];

		return create(newValues);
	}

	@Override
	public MatrixBlockDictionary getMBDict(int nCol) {
		return MatrixBlockDictionary.createDictionary(_values, nCol, true);
	}

	@Override
	public void aggregateCols(double[] c, Builtin fn, int[] colIndexes) {
		final int nCol = colIndexes.length;
		final int rlen = _values.length / nCol;
		for(int k = 0; k < rlen; k++)
			for(int j = 0, valOff = k * nCol; j < nCol; j++)
				c[colIndexes[j]] = fn.execute(c[colIndexes[j]], _values[valOff + j]);
	}

	@Override
	public void aggregateColsWithReference(double[] c, Builtin fn, int[] colIndexes, double[] reference, boolean def) {
		final int nCol = reference.length;
		final int rlen = _values.length / nCol;
		for(int k = 0; k < rlen; k++)
			for(int j = 0, valOff = k * nCol; j < nCol; j++)
				c[colIndexes[j]] = fn.execute(c[colIndexes[j]], _values[valOff + j] + reference[j]);
		if(def)
			for(int i = 0; i < nCol; i++)
				c[colIndexes[i]] = fn.execute(c[colIndexes[i]], reference[i]);
	}

	@Override
	public ADictionary scaleTuples(int[] scaling, int nCol) {
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
	public Dictionary preaggValuesFromDense(int numVals, int[] colIndexes, int[] aggregateColumns, double[] b, int cut) {
		double[] ret = new double[numVals * aggregateColumns.length];
		for(int k = 0, off = 0; k < numVals * colIndexes.length; k += colIndexes.length, off += aggregateColumns.length) {
			for(int h = 0; h < colIndexes.length; h++) {
				int idb = colIndexes[h] * cut;
				double v = _values[k + h];
				if(v != 0)
					for(int i = 0; i < aggregateColumns.length; i++)
						ret[off + i] += v * b[idb + aggregateColumns[i]];
			}
		}
		return create(ret);
	}

	@Override
	public ADictionary replace(double pattern, double replace, int nCol) {
		double[] retV = new double[_values.length];
		for(int i = 0; i < _values.length; i++) {
			final double v = _values[i];
			retV[i] = v == pattern ? replace : v;
		}
		return create(retV);
	}

	@Override
	public ADictionary replaceWithReference(double pattern, double replace, double[] reference) {
		final double[] retV = new double[_values.length];
		final int nCol = reference.length;
		final int nRow = _values.length / nCol;
		int off = 0;
		for(int i = 0; i < nRow; i++) {
			for(int j = 0; j < nCol; j++) {
				final double v = _values[off];
				retV[off++] = v + reference[j] == pattern ? replace - reference[j] : v;

			}
		}
		return create(retV);
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
		final int len = counts.length;
		final int nCol = reference.length;
		BigDecimal tmp = BigDecimal.ONE;
		int off = 0;
		for(int i = 0; i < len; i++) {
			for(int j = 0; j < nCol; j++) {
				final double v = _values[off++] + reference[j];
				if(v == 0) {
					ret[0] = 0;
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
	public CM_COV_Object centralMoment(CM_COV_Object ret, ValueFunction fn, int[] counts, int nRows) {
		// should be guaranteed to only contain one value per tuple in dictionary.
		for(int i = 0; i < _values.length; i++)
			fn.execute(ret, _values[i], counts[i]);

		if(ret.getWeight() < nRows)
			fn.execute(ret, 0, nRows - ret.getWeight());
		return ret;
	}

	@Override
	public CM_COV_Object centralMomentWithDefault(CM_COV_Object ret, ValueFunction fn, int[] counts, double def,
		int nRows) {
		// should be guaranteed to only contain one value per tuple in dictionary.
		for(int i = 0; i < _values.length; i++)
			fn.execute(ret, _values[i], counts[i]);

		if(ret.getWeight() < nRows)
			fn.execute(ret, def, nRows - ret.getWeight());
		return ret;
	}

	@Override
	public CM_COV_Object centralMomentWithReference(CM_COV_Object ret, ValueFunction fn, int[] counts, double reference,
		int nRows) {
		// should be guaranteed to only contain one value per tuple in dictionary.
		for(int i = 0; i < _values.length; i++)
			fn.execute(ret, _values[i] + reference, counts[i]);

		if(ret.getWeight() < nRows)
			fn.execute(ret, reference, nRows - ret.getWeight());
		return ret;
	}

	@Override
	public ADictionary rexpandCols(int max, boolean ignore, boolean cast, int nCol) {
		MatrixBlockDictionary a = getMBDict(nCol);
		if(a == null)
			return null;
		return a.rexpandCols(max, ignore, cast, nCol);
	}

	@Override
	public ADictionary rexpandColsWithReference(int max, boolean ignore, boolean cast, int reference) {
		MatrixBlockDictionary a = getMBDict(1);
		if(a == null)
			a = new MatrixBlockDictionary(new MatrixBlock(_values.length, 1, (double) reference));
		else
			a = (MatrixBlockDictionary) a.applyScalarOp(new LeftScalarOperator(Plus.getPlusFnObject(), reference));
		if(a == null)
			return null;
		return a.rexpandCols(max, ignore, cast, 1);
	}

	@Override
	public double getSparsity() {
		return 1;
	}

	@Override
	public void multiplyScalar(double v, double[] ret, int off, int dictIdx, int[] cols) {
		final int offD = dictIdx * cols.length;
		for(int i = 0; i < cols.length; i++) {
			double a = v * _values[offD + i];
			ret[off + cols[i]] += a;
		}
	}

	@Override
	protected void TSMMWithScaling(int[] counts, int[] rows, int[] cols, MatrixBlock ret) {
		DictLibMatrixMult.TSMMDictsDenseWithScaling(_values, rows, cols, counts, ret);
	}

	@Override
	protected void MMDict(ADictionary right, int[] rowsLeft, int[] colsRight, MatrixBlock result) {
		right.MMDictDense(_values, rowsLeft, colsRight, result);
	}

	@Override
	protected void MMDictDense(double[] left, int[] rowsLeft, int[] colsRight, MatrixBlock result) {
		DictLibMatrixMult.MMDictsDenseDense(left, _values, rowsLeft, colsRight, result);
	}

	@Override
	protected void MMDictSparse(SparseBlock left, int[] rowsLeft, int[] colsRight, MatrixBlock result) {
		DictLibMatrixMult.MMDictsSparseDense(left, _values, rowsLeft, colsRight, result);
	}

	@Override
	protected void TSMMToUpperTriangle(ADictionary right, int[] rowsLeft, int[] colsRight, MatrixBlock result) {
		right.TSMMToUpperTriangleDense(_values, rowsLeft, colsRight, result);
	}

	@Override
	protected void TSMMToUpperTriangleDense(double[] left, int[] rowsLeft, int[] colsRight, MatrixBlock result) {
		DictLibMatrixMult.MMToUpperTriangleDenseDense(left, _values, rowsLeft, colsRight, result);
	}

	@Override
	protected void TSMMToUpperTriangleSparse(SparseBlock left, int[] rowsLeft, int[] colsRight, MatrixBlock result) {
		DictLibMatrixMult.MMToUpperTriangleSparseDense(left, _values, rowsLeft, colsRight, result);
	}

	@Override
	protected void TSMMToUpperTriangleScaling(ADictionary right, int[] rowsLeft, int[] colsRight, int[] scale,
		MatrixBlock result) {
		right.TSMMToUpperTriangleDenseScaling(_values, rowsLeft, colsRight, scale, result);
	}

	@Override
	protected void TSMMToUpperTriangleDenseScaling(double[] left, int[] rowsLeft, int[] colsRight, int[] scale,
		MatrixBlock result) {
		DictLibMatrixMult.TSMMToUpperTriangleDenseDenseScaling(left, _values, rowsLeft, colsRight, scale, result);
	}

	@Override
	protected void TSMMToUpperTriangleSparseScaling(SparseBlock left, int[] rowsLeft, int[] colsRight, int[] scale,
		MatrixBlock result) {
		DictLibMatrixMult.TSMMToUpperTriangleSparseDenseScaling(left, _values, rowsLeft, colsRight, scale, result);
	}

	@Override
	public boolean eq(ADictionary o) {
		if(o instanceof Dictionary)
			return Arrays.equals(_values, ((Dictionary) o)._values);
		else if(o instanceof MatrixBlockDictionary) {
			final MatrixBlock mb = ((MatrixBlockDictionary) o).getMatrixBlock();
			if(mb.isInSparseFormat())
				return mb.getSparseBlock().equals(_values, mb.getNumColumns());
			final double[] dv = mb.getDenseBlockValues();
			return Arrays.equals(_values, dv);
		}
		return false;
	}
}
