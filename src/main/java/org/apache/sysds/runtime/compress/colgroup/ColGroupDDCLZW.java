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
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.ExecutorService;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import org.apache.arrow.vector.complex.writer.BitWriter;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils.P;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IdentityDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.RangeIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffsetIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.colgroup.scheme.DDCScheme;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.jboss.netty.handler.codec.compression.CompressionException;
import shaded.parquet.it.unimi.dsi.fastutil.ints.IntArrayList;
import shaded.parquet.it.unimi.dsi.fastutil.longs.Long2IntLinkedOpenHashMap;


import java.util.Map;
import java.util.HashMap;
import java.util.Stack;

/**
 * Class to encapsulate information about a column group that is encoded with dense dictionary encoding (DDC) whose
 * mapping vector is additionally lzw compressed.
 * Idea:
 * - DDCLZW stores the mapping vector exclusively in compressed form.
 * - No persistent MapToData cache is maintained.
 * - Sequential operations decode on-the-fly, while operations requiring random access explicitly materialize and fall back to DDC.
 */
public class ColGroupDDCLZW extends APreAgg implements IMapToDataGroup {
    private static final long serialVersionUID = -5769772089913918987L;

    private final int[] _dataLZW; // LZW compressed representation of the mapping
    private final int _nRows; // Number of rows in the mapping vector
    private final int _nUnique; // Number of unique values in the mapping vector

    // Builds a packed 64-bit key for (prefixCode(w), nextSymbol(k)) pairs used in the LZW dictionary. (TODO)
    private static long packKey(int prefixCode, int nextSymbol) {
        return (((long) prefixCode) << 32) | (nextSymbol & 0xffffffffL);
    }

    // Compresses a mapping (AMapToData) into an LZW-compressed byte/integer/? array. (TODO)
    private static int[] compress(AMapToData data) {
        if (data == null)
            throw new IllegalArgumentException("Invalid input: data is null");

        final int nRows = data.size();
        if (nRows <= 0) {
            throw new IllegalArgumentException("Invalid input: data has no rows");
        }

        final int nUnique = data.getUnique();
        if (nUnique <= 0) {
            throw new IllegalArgumentException("Invalid input: data has no unique values");
        }

        // Fast-path: single symbol
        if (nRows == 1)
            return new int[]{data.getIndex(0)};


        // LZW dictionary. Maps (prefixCode, nextSymbol) -> newCode (to a new code).
        // Using fastutil keeps lookups fast. (TODO improve time/space complexity)
        final Long2IntLinkedOpenHashMap dict = new Long2IntLinkedOpenHashMap(1 << 16);
        dict.defaultReturnValue(-1);

        // Output buffer (heuristic capacity; avoids frequent reallocs)
        final IntArrayList out = new IntArrayList(Math.max(16, nRows / 2));

        // Codes {0,...,nUnique - 1} are reserved for the original symbols.
        int nextCode = nUnique;

        // Initialize w with the first input symbol.
        // AMapToData stores dictionary indices, not actual data values.
        // Since indices reference positions in an IDictionary, they are always in the valid index range 0 … nUnique−1;
        int w = data.getIndex(0);

        // Process the remaining input symbols.
        // Example: _data = [2,0,2,3,0,2,1,0,2].
        for (int i = 1; i < nRows; i++) {
            final int k = data.getIndex(i); // next input symbol

            if (k < 0 || k >= nUnique)
                throw new IllegalArgumentException("Symbol out of range: " + k + " (nUnique=" + nUnique + ")");

            final long key = packKey(w, k); // encode (w,k) into long key

            int wk = dict.get(key); // look if wk exists in dict
            if (wk != -1) {
                w = wk; // wk exists in dict so replace w by wk and continue.
            } else {
                // wk does not exist in dict. output current phrase, add new phrase, restart at k
                out.add(w);
                dict.put(key, nextCode++);
                w = k; // Start new phrase with k
            }
        }

        out.add(w);
        return out.toIntArray();
    }

    // Unpack upper 32 bits (w) of (w,k) key pair.
    private static int unpackfirst(long key) {
        return (int) (key >>> 32);
    }

    // Unpack lower 32 bits (k) of (w,k) key pair.
    private static int unpacksecond(long key) {
        return (int) (key);
    }

    // Append symbol to end of int-array.
    private static int[] packint(int[] arr, int last) {
        int[] result = Arrays.copyOf(arr, arr.length + 1);
        result[arr.length] = last;
        return result;
    }

    // Reconstruct phrase to lzw-code.
    private static int[] unpack(int code, int nUnique, Map<Integer, Long> dict) {
        // Base symbol (implicit alphabet)
        if (code < nUnique)
            return new int[]{code};

        Stack<Integer> stack = new Stack<>();
        int c = code;

        while (c >= nUnique) {
            Long key = dict.get(c);
            if (key == null)
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
        while (!stack.isEmpty()) {
            outarray[i++] = stack.pop();
        }
        return outarray;
    }

    // Decompresses an LZW-compressed vector into its pre-compressed AMapToData form.
    // TODO: Compatibility with compress() and used data structures. Improve time/space complexity.
    private static AMapToData decompress(int[] codes, int nUnique, int nRows) {
        // Validate input arguments.
        if (codes == null)
            throw new IllegalArgumentException("codes is null");
        if (codes.length == 0)
            throw new IllegalArgumentException("codes is empty");
        if (nUnique <= 0)
            throw new IllegalArgumentException("Invalid alphabet size: " + nUnique);
        if (nRows <= 0) {
            throw new IllegalArgumentException("Invalid nRows: " + nRows);
        }

        // Maps: code -> packKey(prefixCode, lastSymbolOfPhrase).
        // Base symbols (0..nUnique-1) are implicit and not stored here.
        final Map<Integer, Long> dict = new HashMap<>();

        // Output mapping that will be reconstructed.
        AMapToData out = MapToFactory.create(nRows, nUnique);
        int outPos = 0; // Current write position in the output mapping.

        // Decode the first code. The first code always expands to a valid phrase without needing
        // any dictionary entries.
        int old = codes[0];
        int[] oldPhrase = unpack(old, nUnique, dict);
        for (int v : oldPhrase)
            out.set(outPos++, v);

        // Next free dictionary code. Codes 0..nUnique-1 are reserved for base symbols.
        int nextCode = nUnique;

        // Process remaining codes.
        for (int i = 1; i < codes.length; i++) {
            int key = codes[i];

            int[] next;
            if (key < nUnique || dict.containsKey(key)) {
                // Normal case: The code is either a base symbol or already present in the dictionary.
                next = unpack(key, nUnique, dict);
            } else {
                // KwKwK special case: The current code refers to a phrase that is being defined right now.
                // next = oldPhrase + first(oldPhrase).
                int first = oldPhrase[0];
                next = packint(oldPhrase, first);
            }

            // Append the reconstructed phrase to the output mapping.
            for (int v : next) out.set(outPos++, v);

            // Add new phrase to dictionary: nextCode -> (old, firstSymbol(next)).
            int first = next[0];
            dict.put(nextCode++, packKey(old, first));

            // Advance.
            old = key;
            oldPhrase = next;
        }

        // Safety check: decoder must produce exactly nRows symbols.
        if (outPos != nRows)
            throw new IllegalStateException("Decompression length mismatch: got " + outPos + " expected " + nRows);

        // Return the reconstructed mapping.
        return out;
    }

    // Build Constructor: Used when creating a new DDCLZW instance during compression/build time. (TODO)
    private ColGroupDDCLZW(IColIndex colIndexes, IDictionary dict, AMapToData data, int[] cachedCounts) {
        super(colIndexes, dict, cachedCounts);

        // Derive metadadata
        _nRows = data.size();
        _nUnique = dict.getNumberOfValues(colIndexes.size());

        // Compress mapping to LZW
        _dataLZW = compress(data);

        if (CompressedMatrixBlock.debug) {
            if (getNumValues() == 0)
                throw new DMLCompressionException("Invalid construction with empty dictionary");
            if (_nRows == 0)
                throw new DMLCompressionException("Invalid length of the data. is zero");
            if (data.getUnique() != dict.getNumberOfValues(colIndexes.size()))
                throw new DMLCompressionException("Invalid map to dict Map has:" + data.getUnique() + " while dict has "
                        + dict.getNumberOfValues(colIndexes.size()));
            int[] c = getCounts();
            if (c.length != dict.getNumberOfValues(colIndexes.size()))
                throw new DMLCompressionException("Invalid DDC Construction");
            data.verify();
        }
    }

    // Read Constructor: Used when creating this group from a serialized form (e.g., reading a compressed matrix from disk/memory stream). (TODO)
    private ColGroupDDCLZW(IColIndex colIndexes, IDictionary dict, int[] dataLZW, int nRows, int nUnique, int[] cachedCounts) {
        super(colIndexes, dict, cachedCounts);

        _dataLZW = dataLZW;
        _nRows = nRows;
        _nUnique = nUnique;

        if (CompressedMatrixBlock.debug) {
            if (getNumValues() == 0)
                throw new DMLCompressionException("Invalid construction with empty dictionary");
            if (_nRows <= 0)
                throw new DMLCompressionException("Invalid length of the data. is zero");
            if (_nUnique <= dict.getNumberOfValues(colIndexes.size()))
                throw new DMLCompressionException("Invalid map to dict Map has:" + _nUnique + " while dict has "
                        + dict.getNumberOfValues(colIndexes.size()));
            int[] c = getCounts();
            if (c.length != dict.getNumberOfValues(colIndexes.size()))
                throw new DMLCompressionException("Invalid DDC Construction");

            // Optional: validate that decoding works (expensive)
            // AMapToData decoded = decode(_dataLZW, _nRows, _nUnique);
            // decoded.verify();
        }
    }

    // Factory method for creating a column group. (AColGroup g = ColGroupDDCLZW.create(...);)
    public static AColGroup create(IColIndex colIndexes, IDictionary dict, AMapToData data, int[] cachedCounts) {
        if (dict == null)
            return new ColGroupEmpty(colIndexes);
        else if (data.getUnique() == 1)
            return ColGroupConst.create(colIndexes, dict);
        else
            return new ColGroupDDCLZW(colIndexes, dict, data, cachedCounts);
    }

    /*
     * TODO: Operations with complex access patterns shall be uncompressed to ddc format.
     *  ... return ColGroupDDC.create(...,decompress(_dataLZW),...). We need to decide which methods are
     *  suitable for sequential and which arent. those who arent then we shall materialize and fall back to ddc
     * */

    public AColGroup convertToDDC() {
        final AMapToData map = decompress(_dataLZW, _nUnique, _nRows);
        final int[] counts = getCounts(); // may be null depending on your group
        return ColGroupDDC.create(_colIndexes, _dict, map, counts);
    }

    // Deserialize ColGroupDDCLZW object in binary stream.
    public static ColGroupDDCLZW read(DataInput in) throws IOException {
        final IColIndex colIndexes = ColIndexFactory.read(in);
        final IDictionary dict = DictionaryFactory.read(in);

        // Metadata for lzw mapping.
        final int nRows = in.readInt();
        final int nUnique = in.readInt();

        // Read compressed mapping array.
        final int len = in.readInt();
        if (len < 0)
            throw new IOException("Invalid LZW data length: " + len);

        final int[] dataLZW = new int[len];
        for (int i = 0; i < len; i++)
            dataLZW[i] = in.readInt();

        // cachedCounts currently not serialized (mirror ColGroupDDC.read which passes null)
        return new ColGroupDDCLZW(colIndexes, dict, dataLZW, nRows, nUnique, null);
    }

    // Serialize a ColGroupDDC-object into binary stream.
    @Override
    public void write(DataOutput out) throws IOException {
        _colIndexes.write(out);
        _dict.write(out);
        out.writeInt(_nRows);
        out.writeInt(_nUnique);
        out.writeInt(_dataLZW.length);
        for (int i : _dataLZW) out.writeInt(i);
    }

    @Override
    public double getIdx(int r, int colIdx) {
        return 0;
    }

    @Override
    public CompressionType getCompType() {
        return null;
    }

    @Override
    protected ColGroupType getColGroupType() {
        return null;
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
        throw new NotImplementedException();
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
    public AMapToData getMapToData() {
        throw new NotImplementedException(); // or decompress and return data...
    }

    @Override
    public boolean sameIndexStructure(AColGroupCompressed that) {
        return that instanceof ColGroupDDCLZW && ((ColGroupDDCLZW) that)._dataLZW == _dataLZW;
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
    protected void decompressToDenseBlockTransposedSparseDictionary(DenseBlock db, int rl, int ru, SparseBlock dict) {

    }

    @Override
    protected void decompressToDenseBlockTransposedDenseDictionary(DenseBlock db, int rl, int ru, double[] dict) {

    }

    @Override
    protected void decompressToSparseBlockTransposedSparseDictionary(SparseBlockMCSR db, SparseBlock dict, int nColOut) {

    }

    @Override
    protected void decompressToSparseBlockTransposedDenseDictionary(SparseBlockMCSR db, double[] dict, int nColOut) {

    }

    @Override
    protected void decompressToDenseBlockSparseDictionary(DenseBlock db, int rl, int ru, int offR, int offC, SparseBlock sb) {

    }

    @Override
    protected void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC, double[] values) {

    }

    @Override
    protected void decompressToSparseBlockSparseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC, SparseBlock sb) {

    }

    @Override
    protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC, double[] values) {

    }

    @Override
    public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {

    }

    @Override
    public AColGroup scalarOperation(ScalarOperator op) {
        return null;
    }

    @Override
    public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
        return null;
    }

    @Override
    public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
        return null;
    }

    @Override
    public AColGroup sliceRows(int rl, int ru) {
        return null;
    }

    @Override
    public AColGroup unaryOperation(UnaryOperator op) {
        return null;
    }

    @Override
    public AColGroup append(AColGroup g) {
        return null;
    }

    @Override
    protected AColGroup appendNInternal(AColGroup[] groups, int blen, int rlen) {
        return null;
    }

    @Override
    public AColGroup recompress() {
        return null;
    }

    @Override
    public CompressedSizeInfoColGroup getCompressionInfo(int nRow) {
        return null;
    }

    @Override
    protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering) {
        return null;
    }

    @Override
    protected void sparseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {

    }

    @Override
    protected void denseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {

    }

    @Override
    public AColGroup[] splitReshape(int multiplier, int nRow, int nColOrg) {
        return new AColGroup[0];
    }

    @Override
    protected boolean allowShallowIdentityRightMult() {
        return false;
    }

    @Override
    protected AColGroup allocateRightMultiplication(MatrixBlock right, IColIndex colIndexes, IDictionary preAgg) {
        return null;
    }

    @Override
    public void preAggregateDense(MatrixBlock m, double[] preAgg, int rl, int ru, int cl, int cu) {

    }

    @Override
    public void preAggregateSparse(SparseBlock sb, double[] preAgg, int rl, int ru, int cl, int cu) {

    }

    @Override
    protected void preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret) {

    }

    @Override
    protected void preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {

    }

    @Override
    protected void preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {

    }

    @Override
    protected void preAggregateThatRLEStructure(ColGroupRLE that, Dictionary ret) {

    }

    @Override
    public void leftMMIdentityPreAggregateDense(MatrixBlock that, MatrixBlock ret, int rl, int ru, int cl, int cu) {

    }

    @Override
    protected int[] getCounts(int[] out) {
        return new int[0];
    }

    @Override
    protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {

    }

    @Override
    protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {

    }

    @Override
    protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {

    }
}

