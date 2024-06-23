package org.apache.sysds.runtime.transform.encode;

import java.util.Objects;

import org.apache.sysds.runtime.transform.encode.ColumnEncoderBin.BinMethod;

public class CacheKey {
    private final int columnId;
    private final EncoderType encoderType;

    private final BinMethod binMethod;

    public CacheKey(int columnId, EncoderType encoderType) {
        this.columnId = columnId;
        this.encoderType = encoderType;
        if (encoderType == EncoderType.Bin) {
            // throw exception
            this.binMethod = null;
        }
        else {
            this.binMethod = null;
        }
    }
    
    public CacheKey(int columnId, EncoderType encoderType, BinMethod binMethod) {
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
        CacheKey cacheKey = (CacheKey) o;
        return columnId == cacheKey.columnId && encoderType == cacheKey.encoderType && binMethod == cacheKey.binMethod;
    }

    @Override
    public int hashCode() {
        return Objects.hash(columnId, encoderType, binMethod);
    }

    @Override
    public String toString() {
        return "CacheKey{" +
                "columnId=" + columnId +
                ", encoderType=" + encoderType +
                ", binMethod=" + binMethod +
                '}';
    }
}
