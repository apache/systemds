package org.apache.sysds.runtime.io.cog;

public enum SampleFormatDataTypes {
    UNSIGNED_INTEGER(1),
    SIGNED_INTEGER(2),
    FLOATING_POINT(3),
    UNDEFINED(4);

    private final int value;

    SampleFormatDataTypes(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    public static SampleFormatDataTypes valueOf(int value) {
        for (SampleFormatDataTypes dataType : SampleFormatDataTypes.values()) {
            if (dataType.getValue() == value) {
                return dataType;
            }
        }
        return null;
    }
}
