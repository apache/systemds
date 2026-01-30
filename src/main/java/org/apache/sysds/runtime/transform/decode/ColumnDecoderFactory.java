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

package org.apache.sysds.runtime.transform.decode;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.transform.TfUtils;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.apache.sysds.runtime.util.CollectionUtils.except;
import static org.apache.sysds.runtime.util.CollectionUtils.unionDistinct;

/**
 * Factory class for constructing appropriate column decoders from transformation specifications.
 * This class interprets the JSON-based spec (used in transform scripts) and generates a
 * ColumnDecoderComposite which handles decoding for multiple column types:
 *
 * - Recode: Converts IDs back to original categorical values
 * - Bin:    Reconstructs approximate numeric values from bins
 * - Dummycode: Converts one-hot encodings back to categories
 * - PassThrough: For columns that require no decoding
 */
public class ColumnDecoderFactory {
    public enum DecoderType {
        Bin,
        Dummycode,
        PassThrough,
        Recode,
    }

    /**
     * Creates a decoder using the full signature (column names, schema, metadata, etc.).
     * This method parses the spec, determines which transformation applies to which column,
     * and instantiates the corresponding decoder implementations.
     *
     * @param spec     JSON specification of the transform pipeline
     * @param colnames column names in the input
     * @param schema   value types for each column
     * @param meta     metadata frame block containing transformation metadata
     * @return a composite column decoder that handles all specified decoding logic
     */
    public static ColumnDecoder createDecoder(String spec, String[] colnames, ValueType[] schema, FrameBlock meta) {
        return createDecoder(spec, colnames, schema, meta, meta.getNumColumns(), -1, -1);
    }

    public static ColumnDecoder createDecoder(String spec, String[] colnames, ValueType[] schema, FrameBlock meta, int clen) {
        return createDecoder(spec, colnames, schema, meta, clen, -1, -1);
    }

    public static ColumnDecoder createDecoder(String spec, String[] colnames, ValueType[] schema, FrameBlock meta, int minCol, int maxCol) {
        return createDecoder(spec, colnames, schema, meta, meta.getNumColumns(), minCol, maxCol);
    }

    /**
     * Core decoder creation method.
     *
     * @param spec     transform JSON spec
     * @param colnames column names
     * @param schema   value types
     * @param meta     metadata block
     * @param clen     number of columns
     * @param minCol   optional column range lower bound (1-based)
     * @param maxCol   optional column range upper bound (1-based)
     * @return decoder instance (composite or single)
     */
    public static ColumnDecoder createDecoder(String spec, String[] colnames, ValueType[] schema,
                                              FrameBlock meta, int clen, int minCol, int maxCol) {
        ColumnDecoder decoder;

        // tracks current column offset in matrix block
        int currOffset = 0;

        try {
            JSONObject jSpec = new JSONObject(spec);
            List<ColumnDecoder> ldecoders = new ArrayList<>();

            // Get full list of columns [1, 2, ..., clen]
            List<Integer> fullSeq = UtilFunctions.getSeqList(1, clen, 1);

            // Parse column ID lists from spec JSON
            List<Integer> binIDs = TfMetaUtils.parseBinningColIDs(jSpec, colnames, minCol, maxCol);
            List<Integer> rcIDs = Arrays.asList(ArrayUtils.toObject(
                    TfMetaUtils.parseJsonIDList(jSpec, colnames, TfUtils.TfMethod.RECODE.toString(), minCol, maxCol)));
            List<Integer> dcIDs = Arrays.asList(ArrayUtils.toObject(
                    TfMetaUtils.parseJsonIDList(jSpec, colnames, TfUtils.TfMethod.DUMMYCODE.toString(), minCol, maxCol)));

            // Merge dummycode columns into recode set
            rcIDs = unionDistinct(rcIDs, dcIDs);

            // Determine pass-through columns (not in recode or bin sets)
            int len = dcIDs.isEmpty() ? Math.min(meta.getNumColumns(), clen) : meta.getNumColumns();
            List<Integer> ptIDs = except(except(UtilFunctions.getSeqList(1, len, 1), rcIDs), binIDs);

            // Default schema fallback
            if( schema == null ) {
                schema = UtilFunctions.nCopies(len, ValueType.STRING);
                for( Integer col : ptIDs )
                    schema[col-1] = ValueType.FP64;
            }

            // Create per-column decoders
            for (int colID : fullSeq) {
                if (binIDs.contains(colID)) {
                    ColumnDecoder dec = new ColumnDecoderBin(schema[colID - 1], colID - 1, currOffset);
                    ldecoders.add(dec);
                    currOffset += 1;
                }
                else if (dcIDs.contains(colID)) {
                    int numDummy = (int) meta.getColumnMetadata(colID - 1).getNumDistinct();
                    ColumnDecoder dec = new ColumnDecoderDummycode(schema[colID - 1], colID - 1, currOffset);
                    ldecoders.add(dec);
                    currOffset += numDummy;
                }
                else if (rcIDs.contains(colID)) {
                    ColumnDecoder dec = new ColumnDecoderRecode(schema[colID - 1], false, colID - 1, currOffset);
                    ldecoders.add(dec);
                    currOffset += 1;
                }
                else if (ptIDs.contains(colID)) {
                    ColumnDecoder dec = new ColumnDecoderPassThrough(schema[colID - 1], colID - 1,
                            ArrayUtils.toPrimitive(dcIDs.toArray(new Integer[0])), currOffset);
                    ldecoders.add(dec);
                    currOffset += 1;
                }
                else {
                    throw new DMLRuntimeException("Decoder not supported: " + colID);
                }
            }

            // Combine into a single composite decoder
            decoder = new ColumnDecoderComposite(schema, ldecoders);
            decoder.setColnames(colnames);
            decoder.initMetaData(meta);

        } catch (Exception ex) {
            throw new DMLRuntimeException(ex);
        }

        return decoder;
    }

    /**
     * Gets the decoder type enum value for a given decoder instance.
     *
     * @param decoder a column decoder instance
     * @return ordinal of the decoder type
     */
    public static int getDecoderType(ColumnDecoder decoder) {
        if (decoder instanceof ColumnDecoderDummycode)
            return DecoderType.Dummycode.ordinal();
        else if (decoder instanceof ColumnDecoderRecode)
            return DecoderType.Recode.ordinal();
        else if (decoder instanceof ColumnDecoderPassThrough)
            return DecoderType.PassThrough.ordinal();
        throw new DMLRuntimeException("Unsupported decoder type: " + decoder.getClass().getCanonicalName());
    }

    /**
     * Creates an empty decoder instance of the given type.
     * Used primarily for deserialization or instantiation via type ID.
     *
     * @param type the decoder type ordinal
     * @return a new (uninitialized) decoder instance
     */
    public static ColumnDecoder createInstance(int type) {
        DecoderType dtype = DecoderType.values()[type];
        switch (dtype) {
            case Dummycode:
                return new ColumnDecoderDummycode(null, -1, -1);
            case PassThrough:
                return new ColumnDecoderPassThrough(null, -1, null, -1);
            case Recode:
                return new ColumnDecoderRecode(null, false, -1, -1);
            default:
                throw new DMLRuntimeException("Unsupported Decoder Type used: " + dtype);
        }
    }
}
