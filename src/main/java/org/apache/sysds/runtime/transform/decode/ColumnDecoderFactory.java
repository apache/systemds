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

    public static ColumnDecoder createDecoder(String spec, String[] colnames, ValueType[] schema, FrameBlock meta, int minCol, int maxCol) {
        return createDecoder(spec, colnames, schema, meta, meta.getNumColumns(), minCol, maxCol);
    }

    public static ColumnDecoder createDecoder(String spec, String[] colnames, ValueType[] schema,
                                              FrameBlock meta, int clen, int minCol, int maxCol) {
        ColumnDecoder decoder = null;
        int currOffset = 0;

        try {
            JSONObject jSpec = new JSONObject(spec);
            List<ColumnDecoder> ldecoders = new ArrayList<>();

            // 获取各类列索引
            List<Integer> binIDs = TfMetaUtils.parseBinningColIDs(jSpec, colnames, minCol, maxCol);
            List<Integer> recodeIDs = Arrays.asList(ArrayUtils.toObject(
                    TfMetaUtils.parseJsonIDList(jSpec, colnames, TfUtils.TfMethod.RECODE.toString(), minCol, maxCol)));
            List<Integer> dummyIDs = Arrays.asList(ArrayUtils.toObject(
                    TfMetaUtils.parseJsonIDList(jSpec, colnames, TfUtils.TfMethod.DUMMYCODE.toString(), minCol, maxCol)));

            // 注意：dummy 不再参与 recode 解码
            List<Integer> ptIDs = except(except(UtilFunctions.getSeqList(1, clen, 1), recodeIDs), binIDs);
            ptIDs = except(ptIDs, dummyIDs); // dummy 列也不能 pass-through

            if (schema == null) {
                schema = UtilFunctions.nCopies(clen, ValueType.STRING);
                for (Integer col : ptIDs)
                    schema[col - 1] = ValueType.FP64;
            }

            // Bin decoder
            for (int col : binIDs) {
                ldecoders.add(new ColumnDecoderBin(schema[col - 1], col - 1, currOffset));
                currOffset++;
            }

            // Dummycode decoder
            for (int col : dummyIDs) {
                ldecoders.add(new ColumnDecoderDummycode(schema[col - 1], col - 1, currOffset));
                currOffset++;
            }

            // Recode decoder
            for (int col : recodeIDs) {
                if (!dummyIDs.contains(col)) { // 避免 dummy 列重复 recode 解码
                    ldecoders.add(new ColumnDecoderRecode(schema[col - 1], false, col - 1, currOffset));
                    currOffset++;
                }
            }

            // PassThrough decoder
            for (int col : ptIDs) {
                ldecoders.add(new ColumnDecoderPassThrough(schema[col - 1], col - 1,
                        ArrayUtils.toPrimitive(dummyIDs.toArray(new Integer[0])), currOffset));
                currOffset++;
            }

            // Composite
            decoder = new ColumnDecoderComposite(schema, ldecoders);
            decoder.setColnames(colnames);
            decoder.initMetaData(meta);

            // 调试信息
            System.out.println("Creating decoder for spec: " + spec);
            System.out.println("Creating decoder types:");
            for (ColumnDecoder dec : ldecoders) {
                System.out.println(dec.getClass() + " for column ID: " + dec.getColID() + ", offset=" + dec.getColOffset());
            }

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
