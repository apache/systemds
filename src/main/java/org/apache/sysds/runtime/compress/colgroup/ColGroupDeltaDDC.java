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
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DeltaDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.utils.ACount;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.compress.utils.DblArrayCountHashMap;
import org.apache.sysds.runtime.compress.utils.DoubleCountHashMap;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

/**
 * Class to encapsulate information about a column group that is first delta encoded then encoded with dense dictionary
 * encoding (DeltaDDC).
 */
public class ColGroupDeltaDDC extends ColGroupDDC {
	private static final long serialVersionUID = -1045556313148564147L;

	private ColGroupDeltaDDC(IColIndex colIndexes, IDictionary dict, AMapToData data, int[] cachedCounts) {
		super(colIndexes, dict, data, cachedCounts);
		if(CompressedMatrixBlock.debug) {
			if(!(dict instanceof DeltaDictionary))
				throw new DMLCompressionException("DeltaDDC must use DeltaDictionary");
		}
	}

	public static AColGroup create(IColIndex colIndexes, IDictionary dict, AMapToData data, int[] cachedCounts) {
		if(dict == null)
			return new ColGroupEmpty(colIndexes);
		
		if(!(dict instanceof DeltaDictionary))
			throw new DMLCompressionException("ColGroupDeltaDDC must use DeltaDictionary");
		
		if(data.getUnique() == 1) {
			DeltaDictionary deltaDict = (DeltaDictionary) dict;
			double[] values = deltaDict.getValues();
			final int nCol = colIndexes.size();
			boolean allZeros = true;
			for(int i = 0; i < nCol; i++) {
				if(!Util.eq(values[i], 0.0)) {
					allZeros = false;
					break;
				}
			}
			if(allZeros) {
				double[] constValues = new double[nCol];
				System.arraycopy(values, 0, constValues, 0, nCol);
				return ColGroupConst.create(colIndexes, Dictionary.create(constValues));
			}
		}
		
		return new ColGroupDeltaDDC(colIndexes, dict, data, cachedCounts);
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.DeltaDDC;
	}

	@Override
	public ColGroupType getColGroupType() {
		return ColGroupType.DeltaDDC;
	}

	public static ColGroupDeltaDDC read(DataInput in) throws IOException {
		IColIndex cols = ColIndexFactory.read(in);
		IDictionary dict = DictionaryFactory.read(in);
		AMapToData data = MapToFactory.readIn(in);
		return new ColGroupDeltaDDC(cols, dict, data, null);
	}

	@Override
	protected void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values) {
		final int nCol = _colIndexes.size();
		final double[] prevRow = new double[nCol];
		
		if(rl > 0) {
			final int dictIdx0 = _data.getIndex(0);
			final int rowIndex0 = dictIdx0 * nCol;
			for(int j = 0; j < nCol; j++) {
				prevRow[j] = values[rowIndex0 + j];
			}
			for(int i = 1; i < rl; i++) {
				final int dictIdx = _data.getIndex(i);
				final int rowIndex = dictIdx * nCol;
				for(int j = 0; j < nCol; j++) {
					prevRow[j] += values[rowIndex + j];
				}
			}
		}

		if(db.isContiguous() && nCol == db.getDim(1) && offC == 0) {
			final int nColOut = db.getDim(1);
			final double[] c = db.values(0);
			for(int i = rl; i < ru; i++) {
				final int dictIdx = _data.getIndex(i);
				final int rowIndex = dictIdx * nCol;
				final int rowBaseOff = (i + offR) * nColOut;
				
				if(i == 0 && rl == 0) {
					for(int j = 0; j < nCol; j++) {
						final double value = values[rowIndex + j];
						c[rowBaseOff + j] = value;
						prevRow[j] = value;
					}
				}
				else {
					for(int j = 0; j < nCol; j++) {
						final double delta = values[rowIndex + j];
						final double newValue = prevRow[j] + delta;
						c[rowBaseOff + j] = newValue;
						prevRow[j] = newValue;
					}
				}
			}
		}
		else {
			for(int i = rl, offT = rl + offR; i < ru; i++, offT++) {
				final double[] c = db.values(offT);
				final int off = db.pos(offT) + offC;
				final int dictIdx = _data.getIndex(i);
				final int rowIndex = dictIdx * nCol;
				
				if(i == 0 && rl == 0) {
					for(int j = 0; j < nCol; j++) {
						final double value = values[rowIndex + j];
						final int colIdx = _colIndexes.get(j);
						c[off + colIdx] = value;
						prevRow[j] = value;
					}
				}
				else {
					for(int j = 0; j < nCol; j++) {
						final double delta = values[rowIndex + j];
						final double newValue = prevRow[j] + delta;
						final int colIdx = _colIndexes.get(j);
						c[off + colIdx] = newValue;
						prevRow[j] = newValue;
					}
				}
			}
		}
	}

	@Override
	protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		double[] values) {
		final int nCol = _colIndexes.size();
		final double[] prevRow = new double[nCol];
		
		if(rl > 0) {
			final int dictIdx0 = _data.getIndex(0);
			final int rowIndex0 = dictIdx0 * nCol;
			for(int j = 0; j < nCol; j++) {
				prevRow[j] = values[rowIndex0 + j];
			}
			for(int i = 1; i < rl; i++) {
				final int dictIdx = _data.getIndex(i);
				final int rowIndex = dictIdx * nCol;
				for(int j = 0; j < nCol; j++) {
					prevRow[j] += values[rowIndex + j];
				}
			}
		}

		for(int i = rl, offT = rl + offR; i < ru; i++, offT++) {
			final int dictIdx = _data.getIndex(i);
			final int rowIndex = dictIdx * nCol;
			
			if(i == 0 && rl == 0) {
				for(int j = 0; j < nCol; j++) {
					final double value = values[rowIndex + j];
					final int colIdx = _colIndexes.get(j);
					ret.append(offT, colIdx + offC, value);
					prevRow[j] = value;
				}
			}
			else {
				for(int j = 0; j < nCol; j++) {
					final double delta = values[rowIndex + j];
					final double newValue = prevRow[j] + delta;
					final int colIdx = _colIndexes.get(j);
					ret.append(offT, colIdx + offC, newValue);
					prevRow[j] = newValue;
				}
			}
		}
	}

	@Override
	protected void decompressToDenseBlockSparseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		throw new NotImplementedException("Dense block decompression from sparse dictionary for DeltaDDC not yet implemented");
	}

	@Override
	protected void decompressToSparseBlockSparseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		throw new NotImplementedException("Sparse block decompression from sparse dictionary for DeltaDDC not yet implemented");
	}

	@Override
	protected void decompressToDenseBlockTransposedSparseDictionary(DenseBlock db, int rl, int ru, SparseBlock sb) {
		throw new NotImplementedException("Transposed dense block decompression from sparse dictionary for DeltaDDC not yet implemented");
	}

	@Override
	protected void decompressToDenseBlockTransposedDenseDictionary(DenseBlock db, int rl, int ru, double[] dict) {
		throw new NotImplementedException("Transposed dense block decompression from dense dictionary for DeltaDDC not yet implemented");
	}

	@Override
	protected void decompressToSparseBlockTransposedSparseDictionary(SparseBlockMCSR sbr, SparseBlock sb, int nColOut) {
		throw new NotImplementedException("Transposed sparse block decompression from sparse dictionary for DeltaDDC not yet implemented");
	}

	@Override
	protected void decompressToSparseBlockTransposedDenseDictionary(SparseBlockMCSR sbr, double[] dict, int nColOut) {
		throw new NotImplementedException("Transposed sparse block decompression from dense dictionary for DeltaDDC not yet implemented");
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			return super.scalarOperation(op);
		}
		else if(op.fn instanceof Plus || op.fn instanceof Minus) {
			return scalarOperationShift(op);
		}
		else {
			AColGroup ddc = convertToDDC();
			return ddc.scalarOperation(op);
		}
	}

	private AColGroup scalarOperationShift(ScalarOperator op) {
		final int nCol = _colIndexes.size();
		final int id0 = _data.getIndex(0);
		final double[] vals = _dict.getValues();
		final double[] tuple0 = new double[nCol];
		for(int j = 0; j < nCol; j++)
			tuple0[j] = vals[id0 * nCol + j];

		final double[] tupleNew = new double[nCol];
		for(int j = 0; j < nCol; j++)
			tupleNew[j] = op.executeScalar(tuple0[j]);

		int[] counts = getCounts();
		if(counts[id0] == 1) {
			double[] newVals = vals.clone();
			for(int j = 0; j < nCol; j++)
				newVals[id0 * nCol + j] = tupleNew[j];
			return create(_colIndexes, new DeltaDictionary(newVals, nCol), _data, counts);
		}
		else {
			int idNew = -1;
			int nEntries = vals.length / nCol;
			for(int k = 0; k < nEntries; k++) {
				boolean match = true;
				for(int j = 0; j < nCol; j++) {
					if(vals[k * nCol + j] != tupleNew[j]) {
						match = false;
						break;
					}
				}
				if(match) {
					idNew = k;
					break;
				}
			}

			IDictionary newDict = _dict;
			if(idNew == -1) {
				double[] newVals = Arrays.copyOf(vals, vals.length + nCol);
				System.arraycopy(tupleNew, 0, newVals, vals.length, nCol);
				newDict = new DeltaDictionary(newVals, nCol);
				idNew = nEntries;
			}

			AMapToData newData = MapToFactory.create(_data.size(), Math.max(_data.getUpperBoundValue(), idNew) + 1);
			for(int i = 0; i < _data.size(); i++)
				newData.set(i, _data.getIndex(i));
			newData.set(0, idNew);
			
			return create(_colIndexes, newDict, newData, null);
		}
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		AColGroup ddc = convertToDDC();
		return ddc.unaryOperation(op);
	}

	@Override
	public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		throw new NotImplementedException("Left matrix multiplication not supported for DeltaDDC");
	}

	@Override
	public void rightDecompressingMult(MatrixBlock right, MatrixBlock ret, int rl, int ru, int nRows, int crl, int cru) {
		throw new NotImplementedException("Right matrix multiplication not supported for DeltaDDC");
	}

	@Override
	public void preAggregateDense(MatrixBlock m, double[] preAgg, int rl, int ru, int cl, int cu) {
		throw new NotImplementedException("Pre-aggregate dense not supported for DeltaDDC");
	}

	@Override
	public void preAggregateSparse(SparseBlock sb, double[] preAgg, int rl, int ru, int cl, int cu) {
		throw new NotImplementedException("Pre-aggregate sparse not supported for DeltaDDC");
	}

	@Override
	public void preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret) {
		throw new NotImplementedException("Pre-aggregate DDC structure not supported for DeltaDDC");
	}

	@Override
	public void preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
		throw new NotImplementedException("Pre-aggregate SDCZeros structure not supported for DeltaDDC");
	}

	@Override
	public void preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
		throw new NotImplementedException("Pre-aggregate SDCSingleZeros structure not supported for DeltaDDC");
	}

	@Override
	protected void preAggregateThatRLEStructure(ColGroupRLE that, Dictionary ret) {
		throw new NotImplementedException("Pre-aggregate RLE structure not supported for DeltaDDC");
	}

	@Override
	protected double computeMxx(double c, Builtin builtin) {
		throw new NotImplementedException("Compute Min/Max not supported for DeltaDDC");
	}

	@Override
	protected void computeColMxx(double[] c, Builtin builtin) {
		throw new NotImplementedException("Compute Column Min/Max not supported for DeltaDDC");
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		throw new NotImplementedException("Compute Row Min/Max not supported for DeltaDDC");
	}

	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		throw new NotImplementedException("Compute Row Sums not supported for DeltaDDC");
	}

	@Override
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		throw new NotImplementedException("Compute Row Product not supported for DeltaDDC");
	}

	@Override
	public boolean containsValue(double pattern) {
		throw new NotImplementedException("Contains value not supported for DeltaDDC");
	}

	@Override
	public AColGroup append(AColGroup g) {
		throw new NotImplementedException("Append not supported for DeltaDDC");
	}

	@Override
	public AColGroup appendNInternal(AColGroup[] g, int blen, int rlen) {
		throw new NotImplementedException("AppendN not supported for DeltaDDC");
	}

	@Override
	public long getNumberNonZeros(int nRows) {
		long nnz = 0;
		final int nCol = _colIndexes.size();
		final double[] prevRow = new double[nCol];
		
		for(int i = 0; i < nRows; i++) {
			final int dictIdx = _data.getIndex(i);
			final double[] vals = _dict.getValues();
			final int rowIndex = dictIdx * nCol;
			
			if(i == 0) {
				for(int j = 0; j < nCol; j++) {
					double val = vals[rowIndex + j];
					prevRow[j] = val;
					if(val != 0)
						nnz++;
				}
			}
			else {
				for(int j = 0; j < nCol; j++) {
					double val = prevRow[j] + vals[rowIndex + j];
					prevRow[j] = val;
					if(val != 0)
						nnz++;
				}
			}
		}
		return nnz;
	}

	@Override
	public AColGroup sliceRows(int rl, int ru) {
		final int nCol = _colIndexes.size();
		double[] firstRowValues = new double[nCol];
		double[] dictVals = ((DeltaDictionary)_dict).getValues();
		
		for(int i = 0; i <= rl; i++) {
			int dictIdx = _data.getIndex(i);
			int dictOffset = dictIdx * nCol;
			if(i == 0) {
				for(int j = 0; j < nCol; j++) firstRowValues[j] = dictVals[dictOffset + j];
			} else {
				for(int j = 0; j < nCol; j++) firstRowValues[j] += dictVals[dictOffset + j];
			}
		}
		
		int nEntries = dictVals.length / nCol;
		int newId = -1;
		for(int k = 0; k < nEntries; k++) {
			boolean match = true;
			for(int j = 0; j < nCol; j++) {
				if(dictVals[k * nCol + j] != firstRowValues[j]) {
					match = false;
					break;
				}
			}
			if(match) {
				newId = k;
				break;
			}
		}
		
		IDictionary newDict = _dict;
		if(newId == -1) {
			double[] newDictVals = Arrays.copyOf(dictVals, dictVals.length + nCol);
			System.arraycopy(firstRowValues, 0, newDictVals, dictVals.length, nCol);
			newDict = new DeltaDictionary(newDictVals, nCol);
			newId = nEntries;
		}
		
		int numRows = ru - rl;
		AMapToData slicedData = MapToFactory.create(numRows, Math.max(_data.getUpperBoundValue(), newId) + 1);
		for(int i = 0; i < numRows; i++)
			slicedData.set(i, _data.getIndex(rl + i));
		
		slicedData.set(0, newId);
		return ColGroupDeltaDDC.create(_colIndexes, newDict, slicedData, null);
	}

	private AColGroup convertToDDC() {
		final int nCol = _colIndexes.size();
		final int nRow = _data.size();
		double[] values = new double[nRow * nCol];

		double[] prevRow = new double[nCol];
		for(int i = 0; i < nRow; i++) {
			final int dictIdx = _data.getIndex(i);
			final double[] dictVals = _dict.getValues();
			final int rowIndex = dictIdx * nCol;

			for(int j = 0; j < nCol; j++) {
				if(i == 0) {
					prevRow[j] = dictVals[rowIndex + j];
				}
				else {
					prevRow[j] = prevRow[j] + dictVals[rowIndex + j];
				}
				values[i * nCol + j] = prevRow[j];
			}
		}

		return compress(values, _colIndexes);
	}

	private static AColGroup compress(double[] values, IColIndex colIndexes) {
		int nRow = values.length / colIndexes.size();
		int nCol = colIndexes.size();

		if(nCol == 1) {
			DoubleCountHashMap map = new DoubleCountHashMap(16);
			AMapToData mapData = MapToFactory.create(nRow, 256);
			for(int i = 0; i < nRow; i++) {
				int id = map.increment(values[i]);
				if(id >= mapData.getUpperBoundValue()) {
					mapData = mapData.resize(Math.max(mapData.getUpperBoundValue() * 2, id + 1));
				}
				mapData.set(i, id);
			}
			if(map.size() == 1)
				return ColGroupConst.create(colIndexes, Dictionary.create(new double[] {map.getMostFrequent()}));

			IDictionary dict = Dictionary.create(map.getDictionary());
			return ColGroupDDC.create(colIndexes, dict, mapData.resize(map.size()), null);
		}
		else {
			DblArrayCountHashMap map = new DblArrayCountHashMap(16);
			AMapToData mapData = MapToFactory.create(nRow, 256);
			DblArray dblArray = new DblArray(new double[nCol]);
			for(int i = 0; i < nRow; i++) {
				System.arraycopy(values, i * nCol, dblArray.getData(), 0, nCol);
				int id = map.increment(dblArray);
				if(id >= mapData.getUpperBoundValue()) {
					mapData = mapData.resize(Math.max(mapData.getUpperBoundValue() * 2, id + 1));
				}
				mapData.set(i, id);
			}
			if(map.size() == 1) {
				ACount<DblArray>[] counts = map.extractValues();
				return ColGroupConst.create(colIndexes, Dictionary.create(counts[0].key().getData()));
			}

			ACount<DblArray>[] counts = map.extractValues();
			Arrays.sort(counts, Comparator.comparingInt(x -> x.id));
			
			double[] dictValues = new double[counts.length * nCol];
			for(int i = 0; i < counts.length; i++) {
				System.arraycopy(counts[i].key().getData(), 0, dictValues, i * nCol, nCol);
			}

			IDictionary dict = Dictionary.create(dictValues);
			return ColGroupDDC.create(colIndexes, dict, mapData.resize(map.size()), null);
		}
	}
}
