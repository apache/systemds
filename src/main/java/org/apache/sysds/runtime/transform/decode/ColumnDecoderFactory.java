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

public class ColumnDecoderFactory {
    public enum DecoderType {
        Bin,
        Dummycode,
        PassThrough,
        Recode,
    }

    public static ColumnDecoder createDecoder(String spec, String[] colnames, ValueType[] schema, FrameBlock meta) {
        return createDecoder(spec, colnames, schema, meta, meta.getNumColumns(), -1, -1);
    }

    public static ColumnDecoder createDecoder(String spec, String[] colnames, ValueType[] schema, FrameBlock meta, int clen) {
        return createDecoder(spec, colnames, schema, meta, clen, -1, -1);
    }

    public static ColumnDecoder createDecoder(String spec, String[] colnames, ValueType[] schema, FrameBlock meta, int minCol, int maxCol) {
        return createDecoder(spec, colnames, schema, meta, meta.getNumColumns(), minCol, maxCol);
    }

    public static ColumnDecoder createDecoder(String spec, String[] colnames, ValueType[] schema,
                                              FrameBlock meta, int clen, int minCol, int maxCol) {
        ColumnDecoder decoder;
        int currOffset = 0;

        try {
            JSONObject jSpec = new JSONObject(spec);
            List<ColumnDecoder> ldecoders = new ArrayList<>();

            List<Integer> fullSeq = UtilFunctions.getSeqList(1, clen, 1);
            List<Integer> binIDs = TfMetaUtils.parseBinningColIDs(jSpec, colnames, minCol, maxCol);
            List<Integer> recodeIDs = Arrays.asList(ArrayUtils.toObject(
                    TfMetaUtils.parseJsonIDList(jSpec, colnames, TfUtils.TfMethod.RECODE.toString(), minCol, maxCol)));
            List<Integer> dummyIDs = Arrays.asList(ArrayUtils.toObject(
                    TfMetaUtils.parseJsonIDList(jSpec, colnames, TfUtils.TfMethod.DUMMYCODE.toString(), minCol, maxCol)));

            List<Integer> ptIDs = except(except(UtilFunctions.getSeqList(1, clen, 1), recodeIDs), binIDs);
            ptIDs = except(ptIDs, dummyIDs);

            for (int colID : fullSeq) {
                if (binIDs.contains(colID)) {
                    ColumnDecoder dec = new ColumnDecoderBin(schema[colID - 1], colID - 1, currOffset);
                    ldecoders.add(dec);
                    currOffset += 1;
                }
                else if (dummyIDs.contains(colID)) {
                    int numDummy = (int) meta.getColumnMetadata(colID - 1).getNumDistinct();
                    ColumnDecoder dec = new ColumnDecoderDummycode(schema[colID - 1], colID - 1, currOffset);
                    ldecoders.add(dec);
                    currOffset += numDummy;
                }
                else if (recodeIDs.contains(colID)) {
                    ColumnDecoder dec = new ColumnDecoderRecode(schema[colID - 1], false, colID - 1, currOffset);
                    ldecoders.add(dec);
                    currOffset += 1;
                }
                else if (ptIDs.contains(colID)) {
                    ColumnDecoder dec = new ColumnDecoderPassThrough(schema[colID - 1], colID - 1,
                            ArrayUtils.toPrimitive(dummyIDs.toArray(new Integer[0])), currOffset);
                    ldecoders.add(dec);
                    currOffset += 1;
                }
                else {
                    throw new DMLRuntimeException("Decoder not supported: " + colID);
                }
            }


            decoder = new ColumnDecoderComposite(schema, ldecoders);
            decoder.setColnames(colnames);
            decoder.initMetaData(meta);

        } catch (Exception ex) {
            throw new DMLRuntimeException(ex);
        }

        return decoder;
    }

    public static int getDecoderType(ColumnDecoder decoder) {
        if (decoder instanceof ColumnDecoderDummycode)
            return DecoderType.Dummycode.ordinal();
        else if (decoder instanceof ColumnDecoderRecode)
            return DecoderType.Recode.ordinal();
        else if (decoder instanceof ColumnDecoderPassThrough)
            return DecoderType.PassThrough.ordinal();
        throw new DMLRuntimeException("Unsupported decoder type: " + decoder.getClass().getCanonicalName());
    }

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
