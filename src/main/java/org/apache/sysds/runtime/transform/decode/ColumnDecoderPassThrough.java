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

import org.apache.sysds.runtime.frame.data.FrameBlock;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

public class ColumnDecoderPassThrough extends ColumnDecoder {

    private static final long serialVersionUID = -8525203889417422598L;

    //private int[] _dcCols = null;
    //private int[] _srcCols = null;

    protected ColumnDecoderPassThrough(ValueType schema, int ptCols, int[] dcCols, int offset) {
        super(schema, ptCols, offset);
        //_dcCols = dcCols;
    }

    public ColumnDecoderPassThrough() {
        super(null, -1, -1);
    }

    @Override
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out) {
        long p1 = System.nanoTime();
        out.ensureAllocatedColumns(in.getNumRows());
        for (int r = 0; r < in.getNumRows(); r++) {
            out.getColumn(_colID).set(r, in.get(r, _offset));
        }
        //columnDecode(in, out, 0, in.getNumRows());
        long p2 = System.nanoTime();
        System.out.println(this.getClass() + "time: " + (p2 - p1) / 1e6 + " ms");
        return out;
    }


    @Override
    public void columnDecode(MatrixBlock in, FrameBlock out, int rl, int ru) {
        for (int r = rl; r < ru; r++) {
            double val = in.get(r, _offset);
            out.set(r, _colID, UtilFunctions.doubleToObject(_schema, val));
        }
    }

    public ColumnDecoder subRangeDecoder(int colStart, int colEnd, int dummycodedOffset){
        return null;
        //List<Integer> colList = new ArrayList<>();
        //List<Integer> dcList = new ArrayList<>();
        //List<Integer> srcList = new ArrayList<>();
//
        //for (int i = 0; i < _colList.length; i++) {
        //    int colID = _colList[i];
        //    if (colID >= colStart && colID < colEnd) {
        //        colList.add(colID - (colStart - 1));
        //        srcList.add(_srcCols[i] - dummycodedOffset);
        //    }
        //}
//
        //Arrays.stream(_dcCols)
        //        .filter(c -> c >= colStart && c < colEnd)
        //        .forEach(dcList::add);
//
        //if (colList.isEmpty())
        //    return null;
//
        //ColumnDecoderPassThrough dec = new ColumnDecoderPassThrough(
        //        Arrays.copyOfRange(_schema, colStart - 1, colEnd - 1),
        //        colList.stream().mapToInt(i -> i).toArray(),
        //        dcList.stream().mapToInt(i -> i).toArray());
        //dec._srcCols = srcList.stream().mapToInt(i -> i).toArray();
        //return dec;
    }
    @Override
    public void initMetaData(FrameBlock meta) {
        /*
        if (_colList == null)
            return; // nothing to initialize for passthrough columns
        if( _dcCols.length > 0 ) {
            //prepare source column id mapping w/ dummy coding
            _srcCols = new int[_colList.length];
            int ix1 = 0, ix2 = 0, off = 0;
            while( ix1<_colList.length ) {
                if( ix2>=_dcCols.length || _colList[ix1] < _dcCols[ix2] ) {
                    _srcCols[ix1] = _colList[ix1] + off;
                    ix1 ++;
                }
                else { //_colList[ix1] > _dcCols[ix2]
                    ColumnMetadata d =meta.getColumnMetadata()[_dcCols[ix2]-1];
                    off += d.isDefault() ? -1 : d.getNumDistinct() - 1;
                    ix2 ++;
                }
            }
        }
        else {
            //prepare direct source column mapping
            _srcCols = _colList;
        }
        */

    }

    @Override
    public void writeExternal(ObjectOutput os)
            throws IOException
    {
        super.writeExternal(os);
        /*
        os.writeInt(_srcCols.length);
        for(int i = 0; i < _srcCols.length; i++)
            os.writeInt(_srcCols[i]);

        os.writeInt(_dcCols.length);
        for(int i = 0; i < _dcCols.length; i++)
            os.writeInt(_dcCols[i]);

         */
    }

    @Override
    public void readExternal(ObjectInput in)
            throws IOException
    {
        super.readExternal(in);
        /*
        _srcCols = new int[in.readInt()];
        for(int i = 0; i < _srcCols.length; i++)
            _srcCols[i] = in.readInt();

        _dcCols = new int[in.readInt()];
        for(int i = 0; i < _dcCols.length; i++)
            _dcCols[i] = in.readInt();

         */
    }
}
