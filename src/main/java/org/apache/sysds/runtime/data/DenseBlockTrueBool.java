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

package org.apache.sysds.runtime.data;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.common.Warnings;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.util.Arrays;

public class DenseBlockTrueBool extends DenseBlockDRB{

    private static final long serialVersionUID = -6741205568083878338L;
    private boolean[] _data;

    public DenseBlockTrueBool(int[] dims) {
        super(dims);
        reset(_rlen, _odims, 0);
    }

    @Override
    protected void allocateBlock(int bix, int length) {
        _data = new boolean[length];
    }

    public DenseBlockTrueBool(int[] dims, boolean[] data) {
        super(dims);
        _data = data;
    }

    @Override
    public boolean isNumeric() {
        return true;
    }

    @Override
    public boolean isNumeric(ValueType vt) {
        return ValueType.TRUE_BOOLEAN == vt;
    }


    @Override
    public void reset(int rlen, int[] odims, double v){
        reset(rlen, odims, v!= 0);
    }

    public void reset(int rlen, int[] odims, boolean bv) {
        int len = rlen * odims[0];
        if( len > capacity() ) {
            _data = new boolean[len];
            if( bv )
                Arrays.fill(_data, bv);
        }
        else {
            Arrays.fill(_data, 0, len, bv);
        }
        _rlen = rlen;
        _odims = odims;
    }

    @Override
    public void resetNoFill(int rlen, int[] odims){
        int len = rlen * odims[0];
        if( len > capacity() )
            _data = new boolean[len];

        _rlen = rlen;
        _odims = odims;
    }

    @Override
    public long capacity() {
        return (_data!=null) ? _data.length : -1;
    }

    @Override
    protected long computeNnz(int bix, int start, int length) {
        return UtilFunctions.computeNnz(_data, start, length);
    }

    @Override
    public double[] values(int r) {
        double[] ret = getReuseRow(false);
        int ix = pos(r);
        int ncol = _odims[0];
        for(int j=0; j<ncol; j++)
            ret[j] = _data[ix+j] ? 1 : 0;
        return ret;
    }

    @Override
    public double[] valuesAt(int bix) {
        int len = _rlen*_odims[0];
        Warnings.warnFullFP64Conversion(len);
        return DataConverter.toDouble(_data, len);
    }

    @Override
    public int index(int r) {
        return 0;
    }

    @Override
    public void incr(int r, int c) {
        Warnings.warnInvalidBooleanIncrement(1);
        _data[pos(r, c)] = true;
    }

    @Override
    public void incr(int r, int c, double delta) {
        Warnings.warnInvalidBooleanIncrement(delta);
        incr(r,c);
    }

    protected void fillBlock(int fromIndex, int toIndex, boolean v) {
        Arrays.fill(_data, fromIndex, toIndex, v);
    }

    protected void fillBlock(int bix, int fromIndex, int toIndex, boolean v) {
        fillBlock(fromIndex, toIndex, v);
    }
    @Override
    protected void fillBlock(int bix, int fromIndex, int toIndex, double v) {
        fillBlock(fromIndex, toIndex, v!=0);
    }

    protected void setInternal(int ix, boolean v) {
        _data[ix] = v;
    }
    protected void setInternal(int bix, int ix, boolean v) {
        setInternal(ix, v);
    }
    @Override
    protected void setInternal(int bix, int ix, double v) {
        setInternal(ix, v != 0);
    }

    @Override
    public DenseBlock set(String s) {
        Arrays.fill(_data, 0, blockSize() * _odims[0], Boolean.parseBoolean(s));
        return this;
    }

    public DenseBlock set(int r, int c, boolean v) {
        _data[pos(r, c)] = v;
        return this;
    }
    @Override
    public DenseBlock set(int r, int c, double v) {
        _data[pos(r, c)] = v != 0;
        return this;
    }

    public DenseBlock set(DenseBlock db) {
        // ToDo: Performance tests and improvements
        double[] data = db.valuesAt(0);
        for (int i = 0; i < _rlen*_odims[0]; i++) {
            _data[i] = data[i] != 0;
        }
        return this;
    }

    @Override
    public DenseBlock set(int rl, int ru, int cl, int cu, DenseBlock db) {
        //TODO perf computed indexes
        for (int r = rl; r < ru; r++) {
            for (int c = cl; c < cu; c++) {
                int i = r * _odims[0] + c;
                _data[i] = db.get(r - rl, c - cl) != 0;
            }
        }
        return this;
    }


    public DenseBlock set(int r, boolean[] v) {
        int ri = r * _odims[0];
        System.arraycopy(v, 0, _data, ri, v.length);
        return this;
    }

    @Override
    public DenseBlock set(int r, double[] v) {
        int ri = r * _odims[0];
        for (int i = ri; i < ri + v.length; i++) {
            _data[i] = v[i-ri] != 0;
        }
        return this;
    }

    public DenseBlock set(int[] ix, boolean v) {
        _data[pos(ix)] = v;
        return this;
    }
    @Override
    public DenseBlock set(int[] ix, double v) {
        _data[pos(ix)] = v != 0;
        return this;
    }

    @Override
    public DenseBlock set(int[] ix, long v) {
        _data[pos(ix)] = v != 0;
        return this;
    }

    @Override
    public DenseBlock set(int[] ix, String v) {
        _data[pos(ix)] = Boolean.parseBoolean(v);
        return this;
    }

    public boolean getBoolean(int r, int c){
        return _data[pos(r,c)];
    }
    @Override
    public double get(int r, int c) {
        return _data[pos(r, c)] ? 1 : 0;
    }

    @Override
    public double get(int[] ix) {
        return _data[pos(ix)] ? 1 : 0;
    }

    @Override
    public String getString(int[] ix) {
        return String.valueOf(_data[pos(ix)]);
    }

    @Override
    public long getLong(int[] ix) {
        return _data[pos(ix)] ? 1 : 0;
    }
}
