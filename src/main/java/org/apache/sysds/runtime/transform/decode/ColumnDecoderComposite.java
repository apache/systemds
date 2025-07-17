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

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class ColumnDecoderComposite extends ColumnDecoder {
    private static final long serialVersionUID = 5790600547144743716L;

    private List<ColumnDecoder> _decoders = null;
    protected ColumnDecoderComposite(ValueType[] schema, List<ColumnDecoder> decoders) {
        super(schema, null,-1);
        _decoders = decoders;
    }

    //public ColumnDecoderComposite() {
    //    super(null, -1);
    //}

    //private List<MatrixBlock> sliceColumns(MatrixBlock mb, int[] cols) {
    //    List<MatrixBlock> list = new ArrayList<>(cols.length);
    //    for (int col : cols) {
    //        //MatrixBlock ret = new MatrixBlock(mb.getNumRows(), 1, false);
    //        //for (int i = 0; i < mb.getNumRows(); i++) {
    //        //    ret.set(i, 0, mb.get(i, col - 1));
    //        //}
    //        //list.add(ret);
    //        MatrixBlock slice = mb.slice(0, mb.getNumRows() - 1,
    //                col - 1, col - 1, new MatrixBlock());
    //        list.add(slice);
    //    }
    //    return list;
    //}

    @Override
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out) {
        return columnDecode(in, out, 5);
        //out.ensureAllocatedColumns(in.getNumRows());
        //for (ColumnDecoder dec : _decoders) {
        //    List<MatrixBlock> slices = sliceColumns(in, dec.getColList());
        //    for (int c = 0; c < slices.size(); c++) {
        //        ColumnDecoder sub = dec.getColList().length == 1 ? dec :
        //                dec.subRangeDecoder(dec.getColList()[c], dec.getColList()[c] + 1, 0);
        //        if (sub == null)
        //            throw new RuntimeException("Decoder does not support column slicing: " + dec.getClass());
        //        if (sub != dec)
        //            sub._colList = new int[]{dec.getColList()[c]};
        //        sub.columnDecode(slices.get(c), out);
        //    }
        //}
        //return out;
    }

    //@Override
    //public FrameBlock columnDecode(MatrixBlock in, FrameBlock out, final int k) {
    //    final ExecutorService pool = CommonThreadPool.get(k);
    //    out.ensureAllocatedColumns(in.getNumRows());
    //    try{
    //        List<Future<FrameBlock>> tasks = new ArrayList<>();
    //        for (ColumnDecoder dec : _decoders) {
    //            if(dec instanceof ColumnDecoderDummycode) {
    //                tasks.add(pool.submit(() -> dec.columnDecode(in, out)));
    //            }
    //            else {
    //                long t1 = System.nanoTime();
    //                List<MatrixBlock> slices = sliceColumns(in, dec.getColList());
    //                long t2 = System.nanoTime();
    //                System.out.println("slice time: " + (t2 - t1) / 1e6 + " ms");
    //                for (int c = 0; c < slices.size(); c++) {
    //                    ColumnDecoder sub = dec.getColList().length == 1 ? dec :
    //                            dec.subRangeDecoder(dec.getColList()[c], dec.getColList()[c] + 1, 0);
    //                    if (sub != dec)
    //                        sub._colList = new int[]{dec.getColList()[c]};
    //                    int finalC = c;
    //                    tasks.add(pool.submit(() -> sub.columnDecode(slices.get(finalC), out)));
    //                }
    //            }
    //        }
    //        for(Future<?> f : tasks)
    //            f.get();
    //        return out;
    //    }
    //    catch (Exception e) {
    //        throw new RuntimeException(e);
    //    }
    //    finally {
    //        pool.shutdown();
    //    }
    //}


    @Override
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out, final int k) {
        long t3 = System.nanoTime();
        final ExecutorService pool = CommonThreadPool.get(k);
        out.ensureAllocatedColumns(in.getNumRows());

        try {
            List<Future<FrameBlock>> tasks = new ArrayList<>();
            for (ColumnDecoder dec : _decoders) {
                tasks.add(pool.submit(() -> {
                    dec.columnDecode(in, out);
                    return null;
                }));
            }
            for (Future<?> task : tasks)
                task.get();
            long t4 = System.nanoTime();
            System.out.println("ColumnDecoder time: " + (t4 - t3) / 1e6 + " ms");
            return out;
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            pool.shutdown();
        }
    }

//if (dec instanceof ColumnDecoderDummycode) {
    //    // Dummycode is assumed to be thread-safe or handles its own parallelism
    //    //tasks.add(pool.submit(() -> dec.columnDecode(in, new FrameBlock(_schema))));
    //} else if (dec instanceof ColumnDecoderRecode) {
    //    //tasks.add(pool.submit(() -> dec.columnDecode(in, new FrameBlock(_schema))));
    //} else {
    //    for (int c = 0; c < slices.size(); c++) {
    //        final int colIx = dec.getColList()[c];
    //        ColumnDecoder sub = dec.subRangeDecoder(colIx, colIx + 1, 0);
//
    //        if (sub == null)
    //            throw new RuntimeException("Decoder does not support column slicing: " + dec.getClass());
//
    //        final MatrixBlock slice = slices.get(c);
    //        //final int colPos = colIx - 1;
    //        final ValueType vt = _schema;
//
    //        tasks.add(pool.submit(() -> {
    //            FrameBlock partial = new FrameBlock(new ValueType[]{vt});
    //            sub.columnDecode(slice, partial);
    //            return partial;
    //        }));
    //    }
    //}


// Wait for tasks to finish and merge column-wise
//int taskIndex = 0;
//for (ColumnDecoder dec : _decoders) {
//    if (dec instanceof ColumnDecoderDummycode) {
//        FrameBlock partial = tasks.get(taskIndex++).get();
//        for (int i = 0; i < dec.getColList().length; i++) {
//            int outCol = dec.getColList()[i] - 1;
//            out.setColumn(outCol, partial.getColumn(i));
//        }
//    } else {
//        for (int c = 0; c < dec.getColList().length; c++) {
//            FrameBlock partial = tasks.get(taskIndex++).get();
//            int outCol = dec.getColList()[c] - 1;
//            out.setColumn(outCol, partial.getColumn(0));
//        }
//    }
//}

    @Override
    public void columnDecode(MatrixBlock in, FrameBlock out, int rl, int ru) {
        // TODO
        for( ColumnDecoder dec : _decoders )
            dec.columnDecode(in, out, rl, ru);
    }

    @Override
    public ColumnDecoder subRangeDecoder(int colStart, int colEnd, int dummycodedOffset) {
        //Todo
        return null;
        //List<ColumnDecoder> subDecoders = new ArrayList<>();
        //for (ColumnDecoder dec : _decoders) {
        //    ColumnDecoder sub = dec.subRangeDecoder(colStart, colEnd, dummycodedOffset);
        //    if (sub != null)
        //        subDecoders.add(sub);
        //}
        //return new ColumnDecoderComposite(Arrays.copyOfRange(_schema, colStart-1, colEnd-1), subDecoders);
    }

    @Override
    public void updateIndexRanges(long[] beginDims, long[] endDims) {
        for(ColumnDecoder dec : _decoders)
            dec.updateIndexRanges(beginDims, endDims);
    }

    @Override
    public void initMetaData(FrameBlock meta) {
        for( ColumnDecoder decoder : _decoders )
            decoder.initMetaData(meta);
    }

    @Override
    public void writeExternal(ObjectOutput out) throws IOException {
        //TODO
    //    super.writeExternal(out);
    //    out.writeInt(_decoders.size());
    //    out.writeInt(_schema == null ? 0:_schema.length); //write #columns
    //    for(ColumnDecoder decoder : _decoders) {
    //        out.writeByte(ColumnDecoderFactory.getDecoderType(decoder));
    //        decoder.writeExternal(out);
    //    }
    }

    @Override
    public void readExternal(ObjectInput in) throws IOException {
        super.readExternal(in);
        int decodersSize = in.readInt();
        int nCols = in.readInt();
        if (nCols > 0 && decodersSize > nCols*2)
            throw new IOException("Too many decoders");
        _decoders = new ArrayList<>();
        for(int i = 0; i < decodersSize; i++) {
            ColumnDecoder decoder = ColumnDecoderFactory.createInstance(in.readByte());
            decoder.readExternal(in);
            _decoders.add(decoder);
        }
    }
}
