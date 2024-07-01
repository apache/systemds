package org.apache.sysds.runtime.transform.encode;

import java.util.Objects;

import org.apache.sysds.runtime.transform.encode.ColumnEncoderBin.BinMethod;

public class EncodeCacheKey {
    private final int columnId;
    private final EncoderType encoderType;

    private final BinMethod binMethod;

    public EncodeCacheKey(int columnId, EncoderType encoderType) {
        this.columnId = columnId;
        this.encoderType = encoderType;
        if (encoderType == EncoderType.Bin) {
            throw new RuntimeException("Cannot set encoderType to Bin without specifying binMethod");
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
