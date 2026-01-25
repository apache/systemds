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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils.P;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.scheme.DDCLZWScheme;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import shaded.parquet.it.unimi.dsi.fastutil.ints.IntArrayList;
import shaded.parquet.it.unimi.dsi.fastutil.longs.Long2IntLinkedOpenHashMap;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.Stack;
import java.util.HashMap;
import java.util.NoSuchElementException;

/**
 * Class to encapsulate information about a column group that is encoded with dense dictionary encoding (DDC) whose
 * mapping vector is additionally lzw compressed. Idea: - DDCLZW stores the mapping vector exclusively in compressed
 * form. - No persistent MapToData cache is maintained. - Sequential operations decode on-the-fly, while operations
 * requiring random access explicitly materialize and fall back to DDC.
 */
public class ColGroupDDCLZW extends APreAgg implements IMapToDataGroup {
	private static final long serialVersionUID = -5769772089913918987L;

	private final int[] _dataLZW; // LZW compressed representation of the mapping
	private final int _nRows; // Number of rows in the mapping vector
	private final int _nUnique; // Number of unique values in the mapping vector

	/**
	 * Builds a packed 64-bit key from a prefix code and a next symbol, typically used in an LZW dictionary.
	 *
	 * @return a 64-bit packed key representing the (prefixCode, nextSymbol) pair
	 */
	private static long packKey(int prefixCode, int nextSymbol) {
		return (((long) prefixCode) << 32) | (nextSymbol & 0xffffffffL);
	}

	/**
	 * Compresses the given data using a the Lempel-Ziv-Welch (LZW) compression algorithm. The compression is performed
	 * on integer dictionary indices stored in the provided AMapToData object.
	 *
	 * @param data The input data to be compressed, represented as an AMapToData object. The data must not be null, must
	 *             have at least one row, and contain valid dictionary indices.
	 * @return An array of integers representing the compressed data, with each integer corresponding to a dictionary
	 * code.
	 * @throws IllegalArgumentException If the input data is null, has no rows, contains no unique values, or has
	 *                                  symbols that exceed the valid dictionary index range.
	 */
	private static int[] compress(AMapToData data) {
		if(data == null)
			throw new IllegalArgumentException("Invalid input: data is null");

		final int nRows = data.size();
		if(nRows <= 0) {
			throw new IllegalArgumentException("Invalid input: data has no rows");
		}

		final int nUnique = data.getUnique();
		if(nUnique <= 0) {
			throw new IllegalArgumentException("Invalid input: data has no unique values");
		}

		if(nRows == 1)
			return new int[] {data.getIndex(0)};

		final Long2IntLinkedOpenHashMap dict = new Long2IntLinkedOpenHashMap(1 << 16);
		dict.defaultReturnValue(-1);

		final IntArrayList out = new IntArrayList(Math.max(16, nRows / 2));

		int nextCode = nUnique;

		int w = data.getIndex(0);

		for(int i = 1; i < nRows; i++) {
			final int k = data.getIndex(i);

			if(k < 0 || k >= nUnique)
				throw new IllegalArgumentException("Symbol out of range: " + k + " (nUnique=" + nUnique + ")");

			final long key = packKey(w, k);

			int wk = dict.get(key);
			if(wk != -1) {
				w = wk;
			}
			else {
				out.add(w);
				dict.put(key, nextCode++);
				w = k;
			}
		}

		out.add(w);
		return out.toIntArray();
	}

	private static int unpackfirst(long key) {
		return (int) (key >>> 32);
	}

	private static int unpacksecond(long key) {
		return (int) (key);
	}

	private static int[] packint(int[] arr, int last) {
		int[] result = Arrays.copyOf(arr, arr.length + 1);
		result[arr.length] = last;
		return result;
	}

	private static int[] unpack(int code, int nUnique, Map<Integer, Long> dict) {
		if(code < nUnique)
			return new int[] {code};

		Stack<Integer> stack = new Stack<>();
		int c = code;

		while(c >= nUnique) {
			Long key = dict.get(c);
			if(key == null)
				throw new IllegalStateException("Missing dictionary entry for code: " + c);

			int symbol = unpacksecond(key);
			stack.push(symbol);
			c = unpackfirst(key);
		}

		// Basissymbol
		stack.push(c);
		int[] outarray = new int[stack.size()];
		int i = 0;
		// korrekt ins Output schreiben
		while(!stack.isEmpty()) {
			outarray[i++] = stack.pop();
		}
		return outarray;
	}

	// Decompresses an LZW-compressed vector into its pre-compressed AMapToData form until index.
	private static AMapToData decompressFull(int[] codes, int nUnique, int nRows) {
		return decompress(codes, nUnique, nRows, nRows);
	}

	/**
	 * The LZWMappingIterator class is responsible for decoding and iterating through an LZW (Lempel-Ziv-Welch)
	 * compressed mapping. This iterator is primarily used for reconstructing symbols and phrases from the compressed
	 * mapping data, maintaining the internal state of the LZW decompression process.
	 *
	 * The decoding process maintains an LZW dictionary, tracks the current and previous phrases, handles new LZW codes,
	 * and provides methods to retrieve or skip mapping symbols.
	 */
	private final class LZWMappingIterator {
		private final Map<Integer, Long> dict = new HashMap<>(); // LZW-dictionary. Maps code -> (prefixCode, nextSymbol).
		private int lzwIndex = 0; // Current position in the LZW-compressed mapping (_dataLZW).
		private int mapIndex = 0; // Number of mapping symbols returned so far.
		private int nextCode = _nUnique; // Next free LZW code.
		private int[] currentPhrase = null; // Current phrase being decoded from the LZW-compressed mapping.
		private int currentPhraseIndex = 0; // Next position in the current phrase to return.
		private int[] oldPhrase = null; // Previous phrase.
		private int oldCode = -1; // Previous code.

		LZWMappingIterator() {
			lzwIndex = 1; // First code consumed during initialization.
			oldCode = _dataLZW[0]; // Decode the first code into initial phrase.
			oldPhrase = unpack(oldCode, _nUnique, dict);
			currentPhrase = oldPhrase;
			currentPhraseIndex = 0;
			mapIndex = 0; // No mapping symbols have been returned yet.
		}

		// True if there are more mapping symbols to decode.
		boolean hasNext() {
			return mapIndex < _nRows;
		}

		void skip(int k) {
			for(int i = 0; i < k; i++)
				next();
		}

		int next() {
			if(!hasNext())
				throw new NoSuchElementException();

			// If the current phrase still has symbols, return the next symbol from it.
			if(currentPhraseIndex < currentPhrase.length) {
				mapIndex++;
				return currentPhrase[currentPhraseIndex++];
			}

			// Otherwises decode the next code into a new phrase.
			if(lzwIndex >= _dataLZW.length)
				throw new IllegalStateException("Invalid LZW index: " + lzwIndex);

			final int key = _dataLZW[lzwIndex++];

			final int[] next;
			if(key < _nUnique || dict.containsKey(key)) {
				next = unpack(key, _nUnique,
					dict); // Normal case: The code is either a base symbol or already present in the dictionary.
			}
			else {
				next = packint(oldPhrase, oldPhrase[0]); // Special case.
			}

			// Add new phrase to dictionary: nextCode -> (oldCode, firstSymbol(next)).
			dict.put(nextCode++, packKey(oldCode, next[0]));

			// Advance decoder state.
			oldCode = key;
			oldPhrase = next;

			// Start returning symbols from the newly decoded phrase.
			currentPhrase = next;
			currentPhraseIndex = 0;

			mapIndex++;
			return currentPhrase[currentPhraseIndex++];
		}
	}

	/**
	 * Decompresses a sequence of LZW-compressed integer codes into an {@code AMapToData} structure representing the
	 * decompressed data.
	 *
	 * @param codes   an array of integer LZW-codes to be decompressed
	 * @param nUnique the number of unique values in the encoding alphabet
	 * @param nRows   the total number of rows in the data to be decompressed
	 * @param index   the length of the decompressed output
	 * @return an {@code AMapToData} instance containing the decompressed data
	 * @throws IllegalArgumentException if {@code codes} is null, empty, or any input parameter is invalid
	 * @throws IllegalStateException    if the decompression process does not produce the expected length of data
	 */
	private static AMapToData decompress(int[] codes, int nUnique, int nRows, int index) {
		if(codes == null)
			throw new IllegalArgumentException("codes is null");
		if(codes.length == 0)
			throw new IllegalArgumentException("codes is empty");
		if(nUnique <= 0)
			throw new IllegalArgumentException("Invalid alphabet size: " + nUnique);
		if(nRows <= 0) {
			throw new IllegalArgumentException("Invalid nRows: " + nRows);
		}
		if(index > nRows) {
			throw new IllegalArgumentException("Index is larger than Data Length: " + index);
		}

		if(index == 0)
			return MapToFactory.create(0, nUnique);

		final Map<Integer, Long> dict = new HashMap<>();

		AMapToData out = MapToFactory.create(index, nUnique);
		int outPos = 0;

		int old = codes[0];
		int[] oldPhrase = unpack(old, nUnique, dict);

		for(int v : oldPhrase) {
			if(outPos == index)
				break;
			out.set(outPos++, v);
		}

		int nextCode = nUnique;

		for(int i = 1; i < codes.length; i++) {
			int key = codes[i];

			int[] next;
			if(key < nUnique || dict.containsKey(key)) {
				next = unpack(key, nUnique, dict);
			}
			else {
				int first = oldPhrase[0];
				next = packint(oldPhrase, first);
			}

			for(int v : next) {
				if(outPos == index)
					return out;
				out.set(outPos++, v);
			}

			final int first = next[0];
			dict.put(nextCode++, packKey(old, first));

			old = key;
			oldPhrase = next;
		}

		if(outPos != index)
			throw new IllegalStateException("Decompression length mismatch: got " + outPos + " expected " + index);

		return out;
	}

	private ColGroupDDCLZW(IColIndex colIndexes, IDictionary dict, AMapToData data, int[] cachedCounts) {
		super(colIndexes, dict, cachedCounts);

		_nRows = data.size();
		_nUnique = dict.getNumberOfValues(colIndexes.size());

		_dataLZW = compress(data);

		if(CompressedMatrixBlock.debug) {
			if(getNumValues() == 0)
				throw new DMLCompressionException("Invalid construction with empty dictionary");
			if(_nRows == 0)
				throw new DMLCompressionException("Invalid length of the data. is zero");
			if(data.getUnique() != dict.getNumberOfValues(colIndexes.size()))
				throw new DMLCompressionException(
					"Invalid map to dict Map has:" + data.getUnique() + " while dict has " +
						dict.getNumberOfValues(colIndexes.size()));
			int[] c = getCounts();
			if(c.length != dict.getNumberOfValues(colIndexes.size()))
				throw new DMLCompressionException("Invalid DDC Construction");
			data.verify();
		}
	}

	/**
	 * Constructs a ColGroupDDCLZW object, which represents a compressed column group in a matrix using the LZW
	 * compression format. The constructor initializes the internal state, validates the input parameters, and ensures
	 * the compressed data conforms to the expected format and dictionary mapping.
	 *
	 * @param colIndexes   the column indices associated with this column group
	 * @param dict         the dictionary containing unique values associated with the column group
	 * @param dataLZW      the compressed data of the column group using LZW encoding
	 * @param nRows        the number of rows represented by this column group
	 * @param nUnique      the number of unique elements in the data, matching the dictionary size
	 * @param cachedCounts a precomputed array for the count of each value in the data
	 */
	private ColGroupDDCLZW(IColIndex colIndexes, IDictionary dict, int[] dataLZW, int nRows, int nUnique,
		int[] cachedCounts) {
		super(colIndexes, dict, cachedCounts);

		_dataLZW = dataLZW;
		_nRows = nRows;
		_nUnique = nUnique;

		if(CompressedMatrixBlock.debug) {
			if(getNumValues() == 0)
				throw new DMLCompressionException("Invalid construction with empty dictionary");
			if(_nRows <= 0)
				throw new DMLCompressionException("Invalid length of the data. is zero");
			if(_nUnique != dict.getNumberOfValues(colIndexes.size()))
				throw new DMLCompressionException("Invalid map to dict Map has:" + _nUnique + " while dict has " +
					dict.getNumberOfValues(colIndexes.size()));
			int[] c = getCounts();
			if(c.length != dict.getNumberOfValues(colIndexes.size()))
				throw new DMLCompressionException("Invalid DDC Construction");
		}
	}

	public static AColGroup create(IColIndex colIndexes, IDictionary dict, AMapToData data, int[] cachedCounts) {
		if(dict == null)
			return new ColGroupEmpty(colIndexes);
		else if(data.getUnique() == 1)
			return ColGroupConst.create(colIndexes, dict);
		else
			return new ColGroupDDCLZW(colIndexes, dict, data, cachedCounts);
	}

	/**
	 * Converts the current ColGroupDDCLZW instance to a ColGroupDDC instance. The method decompresses the
	 * LZW-compressed data of this instance, reconstructs the mapping to the decompressed data, and creates a new
	 * ColGroupDDC instance with the decompressed mapping, dictionary, and column indexes of this instance.
	 *
	 * @return an AColGroup instance representing the decoded ColGroup in DDC format
	 */
	public AColGroup convertToDDC() {
		final AMapToData map = decompress(_dataLZW, _nUnique, _nRows, _nRows);
		final int[] counts = getCounts(); // may be null depending on your group
		return ColGroupDDC.create(_colIndexes, _dict, map, counts);
	}

	/**
	 * Reads and constructs a ColGroupDDCLZW object from the given DataInput.
	 *
	 * @param in the DataInput to read the group data from
	 * @return a new ColGroupDDCLZW object with the data read from the input
	 * @throws IOException if an I/O error occurs during reading, or if the input data is invalid
	 */
	public static ColGroupDDCLZW read(DataInput in) throws IOException {
		final IColIndex colIndexes = ColIndexFactory.read(in);
		final IDictionary dict = DictionaryFactory.read(in);

		final int nRows = in.readInt();
		final int nUnique = in.readInt();

		final int len = in.readInt();
		if(len < 0)
			throw new IOException("Invalid LZW data length: " + len);

		final int[] dataLZW = new int[len];
		for(int i = 0; i < len; i++)
			dataLZW[i] = in.readInt();

		return new ColGroupDDCLZW(colIndexes, dict, dataLZW, nRows, nUnique, null);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		out.writeInt(_nRows);
		out.writeInt(_nUnique);
		out.writeInt(_dataLZW.length);
		for(int i : _dataLZW)
			out.writeInt(i);
	}

	@Override
	public double getIdx(int r, int colIdx) {
		if(r < 0 || r >= _nRows)
			throw new DMLRuntimeException("Row index out of bounds");

		if(colIdx < 0 || colIdx >= _colIndexes.size())
			throw new DMLRuntimeException("Column index out of bounds");

		final LZWMappingIterator it = new LZWMappingIterator();
		int dictIdx = -1;
		for(int i = 0; i <= r; i++) {
			dictIdx = it.next();
		}
		return _dict.getValue(dictIdx, colIdx, _colIndexes.size());
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.DDCLZW;
	}

	@Override
	protected ColGroupType getColGroupType() {
		return ColGroupType.DDCLZW;
	}

	@Override
	public boolean containsValue(double pattern) {
		return _dict.containsValue(pattern);
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		final int nVals = getNumValues();
		final int nCols = getNumCols();
		return e.getCost(nRows, nRows, nCols, nVals, _dict.getSparsity());
	}

	@Override
	public ICLAScheme getCompressionScheme() {
		return DDCLZWScheme.create(this);
	}

	@Override
	protected int numRowsToMultiply() {
		return _nRows;
	}

	@Override
	protected AColGroup copyAndSet(IColIndex colIndexes, IDictionary newDictionary) {
		return new ColGroupDDCLZW(colIndexes, newDictionary, _dataLZW, _nRows, _nUnique, getCachedCounts());
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += 4; // _nRows size
		ret += 4; // _nUnique size
		ret += 4; // dataLZW.length
		ret += (long) _dataLZW.length * 4; //lzw codes
		return ret;
	}

	@Override
	public AMapToData getMapToData() {
		return decompressFull(_dataLZW, _nUnique, _nRows);
	}

	@Override
	public boolean sameIndexStructure(AColGroupCompressed that) {
		return that instanceof ColGroupDDCLZW && ((ColGroupDDCLZW) that)._dataLZW == this._dataLZW;
	}

	@Override
	protected double computeMxx(double c, Builtin builtin) {
		return _dict.aggregate(c, builtin);
	}

	@Override
	protected void computeColMxx(double[] c, Builtin builtin) {
		_dict.aggregateCols(c, builtin, _colIndexes);
	}

	@Override
	public AColGroup sliceRows(int rl, int ru) {
		try {
			if(rl < 0 || ru > _nRows)
				throw new DMLRuntimeException("Invalid slice range: " + rl + " - " + ru);

			final int len = ru - rl;
			if(len == 0)
				return new ColGroupEmpty(_colIndexes);

			final int[] slicedMapping = new int[len];

			final LZWMappingIterator it = new LZWMappingIterator();

			for(int i = 0; i < rl; i++)
				it.next();

			for(int i = rl; i < ru; i++)
				slicedMapping[i - rl] = it.next();

			AMapToData slicedMappingAMapToData = MapToFactory.create(len, _nUnique);
			for(int i = 0; i < len; i++) {
				slicedMappingAMapToData.set(i, slicedMapping[i]);
			}

			return new ColGroupDDCLZW(_colIndexes, _dict, slicedMappingAMapToData, null);
		}
		catch(Exception e) {
			throw new DMLRuntimeException("Failed to slice out sub part DDCLZW: " + rl + ", " + ru, e);
		}
	}

	@Override
	protected void decompressToDenseBlockTransposedSparseDictionary(DenseBlock db, int rl, int ru, SparseBlock sb) {
		LZWMappingIterator it = new LZWMappingIterator();
		for(int i = 0; i < rl; i++) {
			it.next();
		}

		for(int i = rl; i < ru; i++) {
			final int vr = it.next();
			if(sb.isEmpty(vr))
				continue;
			final int apos = sb.pos(vr);
			final int alen = sb.size(vr) + apos;
			final int[] aix = sb.indexes(vr);
			final double[] aval = sb.values(vr);
			for(int j = apos; j < alen; j++) {
				final int rowOut = _colIndexes.get(aix[j]);
				final double[] c = db.values(rowOut);
				final int off = db.pos(rowOut);
				c[off + i] += aval[j];
			}
		}
	}

	@Override
	protected void decompressToDenseBlockTransposedDenseDictionary(DenseBlock db, int rl, int ru, double[] dict) {
		ColGroupDDC g = (ColGroupDDC) convertToDDC();
		g.decompressToDenseBlockTransposedDenseDictionary(db, rl, ru, dict);

	}

	@Override
	protected void decompressToSparseBlockTransposedSparseDictionary(SparseBlockMCSR sbr, SparseBlock sb, int nColOut) {

		int[] colCounts = _dict.countNNZZeroColumns(getCounts());
		for(int j = 0; j < _colIndexes.size(); j++)
			sbr.allocate(_colIndexes.get(j), colCounts[j]);

		LZWMappingIterator it = new LZWMappingIterator();

		for(int i = 0; i < _nRows; i++) {
			int di = it.next();
			if(sb.isEmpty(di))
				continue;

			final int apos = sb.pos(di);
			final int alen = sb.size(di) + apos;
			final int[] aix = sb.indexes(di);
			final double[] aval = sb.values(di);

			for(int j = apos; j < alen; j++) {
				sbr.append(_colIndexes.get(aix[j]), i, aval[apos]);
			}
		}
	}

	@Override
	protected void decompressToSparseBlockTransposedDenseDictionary(SparseBlockMCSR db, double[] dict, int nColOut) {
		ColGroupDDC g = (ColGroupDDC) convertToDDC();
		g.decompressToSparseBlockTransposedDenseDictionary(db, dict, nColOut); // Possible implementation with iterator.
	}

	@Override
	protected void decompressToDenseBlockSparseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		LZWMappingIterator it = new LZWMappingIterator();
		for(int i = 0; i < rl; i++) {
			it.next(); // Skip to rl.
		}

		for(int r = rl, offT = rl + offR; r < ru; r++, offT++) {
			final int vr = it.next();
			if(sb.isEmpty(vr))
				continue;
			final double[] c = db.values(offT);
			final int off = db.pos(offT) + offC;
			_colIndexes.decompressToDenseFromSparse(sb, vr, off, c);
		}
	}

	@Override
	protected void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values) {
		final int nCol = _colIndexes.size();
		final LZWMappingIterator it = new LZWMappingIterator();

		for(int i = 0; i < rl; i++) {
			it.next();
		}

		if(db.isContiguous() && nCol == db.getDim(1) && offC == 0) {
			final int nColOut = db.getDim(1);
			final double[] c = db.values(0);

			for(int i = rl; i < ru; i++) {
				final int dictIdx = it.next();
				final int rowIndex = dictIdx * nCol;
				final int rowBaseOff = (i + offR) * nColOut;

				for(int j = 0; j < nCol; j++)
					c[rowBaseOff + j] = values[rowIndex + j];
			}
		}
		else {
			for(int i = rl, offT = rl + offR; i < ru; i++, offT++) {
				final double[] c = db.values(offT);
				final int off = db.pos(offT) + offC;
				final int dictIdx = it.next();
				final int rowIndex = dictIdx * nCol;

				for(int j = 0; j < nCol; j++) {
					final int colIdx = _colIndexes.get(j);
					c[off + colIdx] = values[rowIndex + j];
				}
			}
		}
	}

	@Override
	protected void decompressToSparseBlockSparseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		LZWMappingIterator it = new LZWMappingIterator();
		for(int i = 0; i < rl; i++) {
			it.next();
		}

		for(int r = rl, offT = rl + offR; r < ru; r++, offT++) {
			final int vr = it.next();
			if(sb.isEmpty(vr))
				continue;
			final int apos = sb.pos(vr);
			final int alen = sb.size(vr) + apos;
			final int[] aix = sb.indexes(vr);
			final double[] aval = sb.values(vr);
			for(int j = apos; j < alen; j++)
				ret.append(offT, offC + _colIndexes.get(aix[j]), aval[j]);
		}
	}

	@Override
	protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		double[] values) {
		decompressToSparseBlockDenseDictionary(ret, rl, ru, offR, offC, values, _colIndexes.size());
	}

	protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		double[] values, int nCol) {
		LZWMappingIterator it = new LZWMappingIterator();
		for(int i = 0; i < rl; i++) {
			it.next();
		}

		for(int i = rl, offT = rl + offR; i < ru; i++, offT++) {
			final int rowIndex = it.next() * nCol;
			for(int j = 0; j < nCol; j++)
				ret.append(offT, _colIndexes.get(j) + offC, values[rowIndex + j]);
		}
	}

	@Override // TODO: Implement! Pays of with LZW!
	public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		convertToDDC().leftMultByMatrixNoPreAgg(matrix, result, rl, ru, cl, cu); // Fallback to DDC.
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		if((op.fn instanceof Plus || op.fn instanceof Minus)) {
			final double v0 = op.executeScalar(0);
			if(v0 == 0)
				return this;
		}
		return new ColGroupDDCLZW(_colIndexes, _dict.applyScalarOp(op), _dataLZW, _nRows, _nUnique, getCachedCounts());
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		return new ColGroupDDCLZW(_colIndexes, _dict.applyUnaryOp(op), _dataLZW, _nRows, _nUnique, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		IDictionary ret = _dict.binOpLeft(op, v, _colIndexes);
		return new ColGroupDDCLZW(_colIndexes, ret, _dataLZW, _nRows, _nUnique, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		if((op.fn instanceof Plus || op.fn instanceof Minus) && _dict instanceof MatrixBlockDictionary &&
			((MatrixBlockDictionary) _dict).getMatrixBlock().isInSparseFormat()) {
			return convertToDDC().binaryRowOpRight(op, v, isRowSafe);
		}
		final IDictionary ret;
		if(_colIndexes.size() == 1)
			ret = _dict.applyScalarOp(new RightScalarOperator(op.fn, v[_colIndexes.get(0)]));
		else
			ret = _dict.binOpRight(op, v, _colIndexes);
		return new ColGroupDDCLZW(_colIndexes, ret, _dataLZW, _nRows, _nUnique, getCachedCounts());
	}

	@Override
	public AColGroup append(AColGroup g) {
		if(g instanceof ColGroupDDCLZW) {
			if(g.getColIndices().equals(_colIndexes)) {
				ColGroupDDCLZW gDDCLZW = (ColGroupDDCLZW) g;
				if(gDDCLZW._dict.equals(_dict)) {
					if(_nUnique == gDDCLZW._nUnique) {
						int[] mergedMap = new int[this._nRows + gDDCLZW._nRows];

						LZWMappingIterator it = new LZWMappingIterator();
						for(int i = 0; i < this._nRows; i++) {
							mergedMap[i] = it.next();
						}

						LZWMappingIterator gLZWit = gDDCLZW.new LZWMappingIterator();
						for(int i = this._nRows; i < mergedMap.length; i++) {
							mergedMap[i] = gLZWit.next();
						}

						AMapToData mergedDataAMap = MapToFactory.create(mergedMap.length, _nUnique);
						int mergedDataAMapPos = 0;

						for(int j : mergedMap) {
							mergedDataAMap.set(mergedDataAMapPos++, j);
						}

						int[] mergedDataAMapCompressed = compress(mergedDataAMap);

						return new ColGroupDDCLZW(_colIndexes, _dict, mergedDataAMapCompressed, mergedMap.length,
							_nUnique, null);
					}
					else
						LOG.warn("Not same unique values therefore not appending DDCLZW\n" + _nUnique + "\n\n" +
							gDDCLZW._nUnique);
				}
				else
					LOG.warn("Not same Dictionaries therefore not appending DDCLZW\n" + _dict + "\n\n" + gDDCLZW._dict);
			}
			else
				LOG.warn(
					"Not same columns therefore not appending DDCLZW\n" + _colIndexes + "\n\n" + g.getColIndices());
		}
		else
			LOG.warn("Not DDCLZW but " + g.getClass().getSimpleName() + ", therefore not appending DDCLZW");
		return null;
	}

	// TODO: adjust according to contract, "this shall only be appended once".
	@Override
	protected AColGroup appendNInternal(AColGroup[] g, int blen, int rlen) {
		/*throw new NotImplementedException();*/
		int[] mergedMap = new int[rlen];
		int mergedMapPos = 0;

		for(int i = 1; i < g.length; i++) {
			if(!_colIndexes.equals(g[i]._colIndexes)) {
				LOG.warn("Not same columns therefore not appending DDCLZW\n" + _colIndexes + "\n\n" + g[i]._colIndexes);
				return null;
			}

			if(!(g[i] instanceof ColGroupDDCLZW)) {
				LOG.warn("Not DDCLZW but " + g[i].getClass().getSimpleName() + ", therefore not appending DDCLZW");
				return null;
			}

			final ColGroupDDCLZW gDDCLZW = (ColGroupDDCLZW) g[i];
			if(!gDDCLZW._dict.equals(_dict)) {
				LOG.warn("Not same Dictionaries therefore not appending DDCLZW\n" + _dict + "\n\n" + gDDCLZW._dict);
				return null;
			}
			if(!(_nUnique == gDDCLZW._nUnique)) {
				LOG.warn(
					"Not same unique values therefore not appending DDCLZW\n" + _nUnique + "\n\n" + gDDCLZW._nUnique);
				return null;
			}
		}

		for(AColGroup group : g) {
			ColGroupDDCLZW gDDCLZW = (ColGroupDDCLZW) group;

			LZWMappingIterator gLZWit = gDDCLZW.new LZWMappingIterator();
			for(int j = 0; j < gDDCLZW._nRows; j++)
				mergedMap[mergedMapPos++] = gLZWit.next();
		}

		AMapToData mergedDataAMap = MapToFactory.create(rlen, _nUnique);
		int mergedDataAMapPos = 0;

		for(int k = 0; k < rlen; k++) {
			mergedDataAMap.set(k, mergedMap[k]);
		}

		int[] mergedDataAMapCompressed = compress(mergedDataAMap);

		return new ColGroupDDCLZW(_colIndexes, _dict, mergedDataAMapCompressed, rlen, _nUnique, null);
	}

	@Override
	public AColGroup recompress() {
		return this; // A new or the same column group depending on optimization goal. (Description DDC)
	}

	@Override
	public CompressedSizeInfoColGroup getCompressionInfo(int nRow) {
		try {
			IEncode enc = getEncoding();
			EstimationFactors ef = new EstimationFactors(_nUnique, _nRows, _nRows, _dict.getSparsity());
			return new CompressedSizeInfoColGroup(_colIndexes, ef, estimateInMemorySize(), getCompType(), enc);
		}
		catch(Exception e) {
			throw new DMLCompressionException(this.toString(), e);
		}
	}

	@Override
	protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering) {
		return new ColGroupDDCLZW(newColIndex, _dict.reorder(reordering), _dataLZW, _nRows, _nUnique,
			getCachedCounts());
	}

	@Override // Correct ?
	public void sparseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		final SparseBlock sb = selection.getSparseBlock();
		final SparseBlock retB = ret.getSparseBlock();
		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;
			final int sPos = sb.pos(r);
			final int rowCompressed = sb.indexes(r)[sPos]; // column index with 1
			decompressToSparseBlock(retB, rowCompressed, rowCompressed + 1, r - rowCompressed, 0);
		}
	}

	@Override // Correct ?
	protected void denseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		// morph(CompressionType.UNCOMPRESSED, _data.size()).sparseSelection(selection, ret, rl, ru);;
		final SparseBlock sb = selection.getSparseBlock();
		final DenseBlock retB = ret.getDenseBlock();
		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;
			final int sPos = sb.pos(r);
			final int rowCompressed = sb.indexes(r)[sPos]; // column index with 1
			decompressToDenseBlock(retB, rowCompressed, rowCompressed + 1, r - rowCompressed, 0);
		}
	}

	@Override
	public AColGroup[] splitReshape(int multiplier, int nRow, int nColOrg) {
		ColGroupDDC g = (ColGroupDDC) convertToDDC();
		return g.splitReshape(multiplier, nRow, nColOrg); // Fallback to ddc. No splitReshapeDDCLZW implemented.
	}

	// Not sure here.
	@Override
	protected boolean allowShallowIdentityRightMult() {
		throw new NotImplementedException();
	}

	@Override
	protected AColGroup allocateRightMultiplication(MatrixBlock right, IColIndex colIndexes, IDictionary preAgg) {
		if(preAgg == null)
			return null;
		else
			return new ColGroupDDCLZW(colIndexes, preAgg, _dataLZW, _nRows, _nUnique, getCachedCounts());
	}

	@Override
	public void preAggregateDense(MatrixBlock m, double[] preAgg, int rl, int ru, int cl, int cu) {
		ColGroupDDC g = (ColGroupDDC) convertToDDC();
		g.preAggregateDense(m, preAgg, rl, ru, cl, cu); // Fallback to ddc.
	}

	@Override
	public void preAggregateSparse(SparseBlock sb, double[] preAgg, int rl, int ru, int cl, int cu) {
		ColGroupDDC g = (ColGroupDDC) convertToDDC();
		g.preAggregateSparse(sb, preAgg, rl, ru, cl, cu); // Fallback to ddc.
	}

	@Override
	protected void preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret) {
		ColGroupDDC g = (ColGroupDDC) convertToDDC();
		g.preAggregateThatDDCStructure(that, ret); // Fallback to ddc.
	}

	@Override
	protected void preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
		ColGroupDDC g = (ColGroupDDC) convertToDDC();
		g.preAggregateThatSDCZerosStructure(that, ret); // Fallback to ddc.
	}

	@Override
	protected void preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
		ColGroupDDC g = (ColGroupDDC) convertToDDC();
		g.preAggregateThatSDCSingleZerosStructure(that, ret); // Fallback to ddc.

	}

	@Override
	protected void preAggregateThatRLEStructure(ColGroupRLE that, Dictionary ret) {
		ColGroupDDC g = (ColGroupDDC) convertToDDC();
		g.preAggregateThatRLEStructure(that, ret); // Fallback to ddc.

	}

	@Override
	public void leftMMIdentityPreAggregateDense(MatrixBlock that, MatrixBlock ret, int rl, int ru, int cl, int cu) {
		ColGroupDDC g = (ColGroupDDC) convertToDDC();
		g.leftMMIdentityPreAggregateDense(that, ret, rl, ru, cl, cu); // Fallback to ddc.
	}

	@Override
	protected int[] getCounts(int[] out) {
		AMapToData data = decompressFull(_dataLZW, _nUnique, _nRows);
		return data.getCounts();
	}

	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		final LZWMappingIterator it = new LZWMappingIterator();
		for(int i = 0; i < rl; i++)
			it.next();

		for(int rix = rl; rix < ru; rix++)
			c[rix] += preAgg[it.next()];
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		final LZWMappingIterator it = new LZWMappingIterator();
		for(int i = 0; i < rl; i++)
			it.next();

		for(int i = rl; i < ru; i++)
			c[i] = builtin.execute(c[i], preAgg[it.next()]);
	}

	@Override
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		final LZWMappingIterator it = new LZWMappingIterator();
		for(int i = 0; i < rl; i++)
			it.next();

		for(int rix = rl; rix < ru; rix++)
			c[rix] *= preAgg[it.next()];
	}
}
