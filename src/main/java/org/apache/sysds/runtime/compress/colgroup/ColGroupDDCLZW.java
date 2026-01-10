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

        // Extract _data values as int array.
        final int[] dataIntVals = new int[nRows];
        for (int i = 0; i < nRows; i++) {
            dataIntVals[i] = data.getIndex(i);
        }

        // Output buffer.
        IntArrayList out = new IntArrayList();
        out.add(nUnique);


        // LZW dictionary. Maps (prefixCode, nextSymbol) to a new code.
        // Using fastutil keeps lookups fast. (TODO Dictionary)
        final Long2IntLinkedOpenHashMap dict = new Long2IntLinkedOpenHashMap(1 << 16);
        dict.defaultReturnValue(-1);

        // Befüllen des Dictionary
        // Abspeichern der Symbole im Output stream
        int index = 0;
        for (int i = 0; i < nRows; i++) {
            if (index == nUnique){
                break;
            }
            int ct = dict.get(dataIntVals[i]);
            if  (ct == -1) {
                dict.put(dataIntVals[i], index++);
                out.add(dataIntVals[i]);
            }
        }
        if (index != nUnique) {
            throw new IllegalArgumentException("Not enough symbols found for number of unique values");
        }

        // Codes {0,...,nUnique - 1} are reserved for the original symbols.
        int nextCode = nUnique;

        // Initialize w with the first input symbol.
        int w = data.getIndex(0);

        // Process the remaining input symbols.
        for (int i = 1; i < nRows; i++) {
            int k = data.getIndex(i); // next input symbol
            long key = packKey(w, k); // encode (w,k) into long key

            int wk = dict.get(key); // look if wk exists in dict
            if (wk != -1) {
                w = wk; // wk exists in dict so replace w by wk and continue.
            } else {
                // wk does not exist in dict.
                out.add(w);
                dict.put(key, nextCode++);
                w = k; // Start new phrase with k
            }
        }

        out.add(w);
        return out.toIntArray();
    }

    private static int unpackfirst(long key){
        return (int)(key >>> 32);
    }

    private static int unpacksecond(long key){
        return (int)(key);
    }

    // Decompresses an LZW-compressed vector into its pre-compressed AMapToData form. (TODO)
    private static int[] packint(int[] arr, int last){
        int[] result = Arrays.copyOf(arr, arr.length+1);
        result[arr.length] = last;
        return result;
    }

    private static int[] unpack(int code, int alphabetSize, Map<Integer, Long> dict) {

        Stack<Integer> stack = new Stack<>();

        int c = code;

        while (c >= alphabetSize) {
            long key = dict.get(c);
            int symbol = unpacksecond(key);
            stack.push(symbol);
            c = unpackfirst(key);
        }

        // Basissymbol
        stack.push(c);
        int [] outarray = new int[stack.size()];
        int i = 0;
        // korrekt ins Output schreiben
        while (!stack.isEmpty()) {
            outarray[i++] = stack.pop();
        }
        return outarray;
    }

    private static void addtoOutput(IntArrayList outarray, int[] code) {
        for (int i = 0; i < code.length; i++) {
            outarray.add(code[i]);
        }
    }

    private static IntArrayList decompress(int[] code) { //TODO: return AMapToData
        // Dictionary
        Map<Integer, Long> dict = new HashMap<>();

        // Extract alphabet size
        int alphabetSize = code[0];


        // Dictionary Initalisierung
        for (int i = 0; i < alphabetSize; i++) {
            dict.put(i, packKey(-1, code[i]));
        }

        // Result der Decompression
        IntArrayList o = new IntArrayList();

        // Decompression
        int old = code[1+alphabetSize];
        int[] next = unpack(old, alphabetSize, dict);
        addtoOutput(o, next);
        int c = next[0];


        for (int i = alphabetSize+2; i < code.length; i++) {
            int key = code[i];
            if (! dict.containsKey(key)) {
                int[] oldnext = unpack(old, alphabetSize, dict);
                int first = oldnext[0];
                next = packint(oldnext, first);
            } else {
                next = unpack(key, alphabetSize, dict);
            }
            for (int inh : next){ // TODO: effizienz
                o.add(inh);
            }
            int first = next[0];
            long s = packKey(old, first);
            dict.put(alphabetSize+i, s);
            old = key;
        }
        return o;
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
}

