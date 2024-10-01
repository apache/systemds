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

import java.util.Objects;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderBin.BinMethod;

public class EncodeCacheKey {
    private final int columnId;
    private final EncoderType encoderType;

    private final BinMethod binMethod;

    public EncodeCacheKey(int columnId, EncoderType encoderType) {
        this.columnId = columnId;
        this.encoderType = encoderType;
        if (encoderType == EncoderType.Bin) {
            throw new DMLRuntimeException("Cannot set encoderType to Bin without specifying binMethod");
        }
        else {
            this.binMethod = null;
        }
    }
    
    public EncodeCacheKey(int columnId, EncoderType encoderType, BinMethod binMethod) {
        this.columnId = columnId;
        this.encoderType = encoderType;
        this.binMethod = binMethod;
    }

    public int getColumnId() {
        return columnId;
    }

    public EncoderType getEncoderType() {
        return encoderType;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        EncodeCacheKey encodeCacheKey = (EncodeCacheKey) o;
        return columnId == encodeCacheKey.columnId && encoderType == encodeCacheKey.encoderType && binMethod == encodeCacheKey.binMethod;
    }

    @Override
    public int hashCode() {
        return Objects.hash(columnId, encoderType, binMethod);
    }

    @Override
    public String toString() {
        return "EncodeCacheKey{" +
                "columnId=" + columnId +
                ", encoderType=" + encoderType +
                ", binMethod=" + binMethod +
                '}';
    }
}
