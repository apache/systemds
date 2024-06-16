package org.apache.sysds.runtime.transform.encode;

import java.util.Objects;

public class CacheKey {
    private final int columnId;
    private final EncoderType encoderType;

    public CacheKey(int columnId, EncoderType encoderType) {
        this.columnId = columnId;
        this.encoderType = encoderType;
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
        return columnId == cacheKey.columnId && encoderType == cacheKey.encoderType;
    }

    @Override
    public int hashCode() {
        return Objects.hash(columnId, encoderType);
    }

    @Override
    public String toString() {
        return "CacheKey{" +
                "columnId=" + columnId +
                ", encoderType=" + encoderType +
                '}';
    }
}
