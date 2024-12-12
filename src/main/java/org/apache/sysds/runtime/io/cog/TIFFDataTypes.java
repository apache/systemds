package org.apache.sysds.runtime.io.cog;

public enum TIFFDataTypes {
    BYTE(1),
    ASCII(2),
    SHORT(3),
    LONG(4),
    RATIONAL(5),
    SBYTE(6),
    UNDEFINED(7),
    SSHORT(8),
    SLONG(9),
    SRATIONAL(10),
    FLOAT(11),
    DOUBLE(12);

    private final int value;

    TIFFDataTypes(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    public int getSize() {
        switch(this) {
            case BYTE:
            case ASCII:
            case SBYTE:
            case UNDEFINED:
                return 1;
            case SHORT:
            case SSHORT:
                return 2;
            case LONG:
            case SLONG:
            case FLOAT:
                return 4;
            case RATIONAL:
            case SRATIONAL:
            case DOUBLE:
                return 8;
            default:
                return 0;
        }
    }

    public static TIFFDataTypes valueOf(int value) {
        for (TIFFDataTypes tag : TIFFDataTypes.values()) {
            if (tag.getValue() == value) {
                return tag;
            }
        }
        return null;
    }
}
