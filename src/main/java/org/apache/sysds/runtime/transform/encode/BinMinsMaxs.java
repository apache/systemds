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

package org.apache.sysds.runtime.transform.encode;

import java.util.Arrays;

public class BinMinsMaxs {

    private final double[] _binMins;

    private final double[] _binMaxs;

    private final long _size;

    public static BinMinsMaxs create(final double[] binMins, final double[] binMaxs) {
        if (binMins.length != binMaxs.length) {
            throw new IllegalArgumentException("Binmins and binmaxs must be of the same length");
        }
        long size = computeSize(binMins, binMaxs);
        return new BinMinsMaxs(binMins, binMaxs, size);
    }

    private BinMinsMaxs(final double[] binMins, final double[] binMaxs, final long size) {
        this._binMins = binMins;
        this._binMaxs = binMaxs;
        this._size = size;
    }

    public double[] get_binMaxs() {
        return this._binMaxs;
    }

    public double[] get_binMins() {
        return this._binMins;
    }

    public long getSize(){
        return this._size;
    }

    private static long computeSize(final double[] binMins, final double[] binMaxs){
        // object header: 16 bytes, reference to _binMins, _binMaxs: each 4 bytes,
        // array header: each 12 bytes, storage of array length: each 4 bytes
        // the constant object overhead is a minimum estimation as exact storage behaviour cannot be predicted
        int size = 56;
        if (binMins != null && binMaxs != null) {
            size += 2 *  8 * binMins.length; // size of double: 8 bytes
        }
        return size;
    }

    @Override
    public String toString() {
        return "BinMinsMaxs{" +
                "_binMins=" + Arrays.toString(_binMins) +
                ", _binMaxs=" + Arrays.toString(_binMaxs) +
                ", _size=" + _size +
                '}';
    }
}
