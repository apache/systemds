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

    public static ColumnDecoder createDecoder(String spec, String[] colnames, ValueType[] schema, FrameBlock meta, int minCol,
                                        int maxCol) {
        return createDecoder(spec, colnames, schema, meta, meta.getNumColumns(), minCol, maxCol);
    }

    public static ColumnDecoder createDecoder(String spec, String[] colnames, ValueType[] schema,
                                        FrameBlock meta, int clen, int minCol, int maxCol)
    {
        ColumnDecoder decoder = null;
        int currOffset = 0;

        try {
            //parse transform specification
            JSONObject jSpec = new JSONObject(spec);
            List<ColumnDecoder> ldecoders = new ArrayList<>();

            //create decoders 'bin', 'recode', 'dummy' and 'pass-through'
            List<Integer> binIDs = TfMetaUtils.parseBinningColIDs(jSpec, colnames, minCol, maxCol);
            List<Integer> rcIDs = Arrays.asList(ArrayUtils.toObject(
                    TfMetaUtils.parseJsonIDList(jSpec, colnames, TfUtils.TfMethod.RECODE.toString(), minCol, maxCol)));
            List<Integer> dcIDs = Arrays.asList(ArrayUtils.toObject(
                    TfMetaUtils.parseJsonIDList(jSpec, colnames, TfUtils.TfMethod.DUMMYCODE.toString(), minCol, maxCol)));
            rcIDs = unionDistinct(rcIDs, dcIDs);
            int len = dcIDs.isEmpty() ? Math.min(meta.getNumColumns(), clen) : meta.getNumColumns();
            List<Integer> ptIDs = except(except(UtilFunctions.getSeqList(1, len, 1), rcIDs), binIDs);

            //create default schema if unspecified (with double columns for pass-through)
            if( schema == null ) {
                schema = UtilFunctions.nCopies(len, ValueType.STRING);
                for( Integer col : ptIDs )
                    schema[col-1] = ValueType.FP64;
            }

            if( !binIDs.isEmpty() ) {
                for (int col : binIDs) {
                    ldecoders.add(new ColumnDecoderBin(schema[col - 1], col - 1, currOffset));
                    currOffset++;
                }
            }
            if( !dcIDs.isEmpty() ) {
                ldecoders.add(new ColumnDecoderDummycode(schema,
                        ArrayUtils.toPrimitive(dcIDs.toArray(new Integer[0])),currOffset));
            }
            if( !rcIDs.isEmpty() ) {
                for( int col : rcIDs ) {
                    ldecoders.add(new ColumnDecoderRecode(schema[col-1], !dcIDs.isEmpty(), col-1, currOffset));
                    currOffset++;
                }
            }
            if( !ptIDs.isEmpty() ) {
                for (int col : ptIDs) {
                    ldecoders.add(new ColumnDecoderPassThrough(schema[col - 1], col - 1,
                            ArrayUtils.toPrimitive(dcIDs.toArray(new Integer[0])), currOffset));
                }
            }

            //create composite decoder of all created decoders
            //and initialize with given meta data (recode, dummy, bin)
            decoder = new ColumnDecoderComposite(schema, ldecoders);
            decoder.setColnames(colnames);
            decoder.initMetaData(meta);
        }
        catch(Exception ex) {
            throw new DMLRuntimeException(ex);
        }

        return decoder;
    }

    public static int getDecoderType(ColumnDecoder decoder) {
        if( decoder instanceof ColumnDecoderDummycode )
            return ColumnDecoderFactory.DecoderType.Dummycode.ordinal();
        else if( decoder instanceof ColumnDecoderRecode )
            return ColumnDecoderFactory.DecoderType.Recode.ordinal();
        else if( decoder instanceof ColumnDecoderPassThrough )
            return ColumnDecoderFactory.DecoderType.PassThrough.ordinal();
        throw new DMLRuntimeException("Unsupported decoder type: "
                + decoder.getClass().getCanonicalName());
    }

    public static ColumnDecoder createInstance(int type) {
        ColumnDecoderFactory.DecoderType dtype = ColumnDecoderFactory.DecoderType.values()[type];

        // create instance
        switch(dtype) {
            case Dummycode:   return new ColumnDecoderDummycode(null, null, -1);
            case PassThrough: return new ColumnDecoderPassThrough(null, -1, null, -1);
            case Recode:      return new ColumnDecoderRecode(null, false, -1, -1);
            default:
                throw new DMLRuntimeException("Unsupported Encoder Type used:  " + dtype);
        }
    }
}
