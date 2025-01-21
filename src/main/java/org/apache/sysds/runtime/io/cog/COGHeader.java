package org.apache.sysds.runtime.io.cog;

import java.util.ArrayList;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class COGHeader {
    private boolean isLittleEndian;
    private String GDALMetadata;
    private IFDTag[] IFD;
    // Do we even need this or will we throw it away?
    // If we keep this and write it again, we also need to write the additional images
    // So this could very likely not make the cut
    private ArrayList<IFDTag[]> additionalIFDs;

    public COGHeader(boolean isLittleEndian) {
        this.isLittleEndian = isLittleEndian;
        GDALMetadata = "";
        additionalIFDs = new ArrayList<IFDTag[]>();
    }

    public void setIFD(IFDTag[] IFD) {
        this.IFD = IFD;
    }

    public IFDTag[] getIFD() {
        return IFD;
    }

    public void addAdditionalIFD(IFDTag[] IFD) {
        additionalIFDs.add(IFD);
    }

    public ArrayList<IFDTag[]> getAdditionalIFDs() {
        return additionalIFDs;
    }

    public IFDTag[] getSingleAdditionalIFD(int index) {
        return additionalIFDs.get(index);
    }

    public void setSingleAdditionalIFD(int index, IFDTag[] IFD) {
        additionalIFDs.set(index, IFD);
    }

    public void removeSingleAdditionalIFD(int index) {
        additionalIFDs.remove(index);
    }

    public void setLittleEndian(boolean isLittleEndian) {
        this.isLittleEndian = isLittleEndian;
    }

    public boolean isLittleEndian() {
        return isLittleEndian;
    }

    public void setGDALMetadata(String GDALMetadata) {
        this.GDALMetadata = GDALMetadata;
    }

    public String getGDALMetadata() {
        return GDALMetadata;
    }

    /**
     * Parses a byte array into a generic number. Can be byte, short, int, float or double
     * depending on the options given. E.g.: Use .doubleValue() on the result to get a double value easily
     *
     * Supported lengths:
     * isDecimal:
     * - 4 bytes: float
     * - 8 bytes: double
     * otherwise:
     * - 1 byte: byte
     * - 2 bytes: short
     * - 4 bytes: int
     * Anything else will throw an exception
     * @param bytes
     * @param length number of bytes that should be read
     * @param offset from the start of the byte array
     * @param isDecimal Whether we are dealing with a floating point number
     * @param isSigned Whether the number is signed
     * @param isRational Whether the number is a rational number as specified in the TIFF standard
     *                   (first 32 bit integer numerator of a fraction, second 32 bit integer denominator)
     * @return
     */
    public Number parseByteArray(byte[] bytes, int length, int offset, boolean isDecimal, boolean isSigned, boolean isRational) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        buffer.order(isLittleEndian ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN);
        buffer.position(offset);

        if (isRational && !isSigned) {
            long numerator = Integer.toUnsignedLong(buffer.getInt());
            long denominator = Integer.toUnsignedLong(buffer.getInt());
            return (double)numerator / denominator;
        }
        if (isRational && isSigned) {
            long numerator = buffer.getInt();
            long denominator = buffer.getInt();
            return (double)numerator / denominator;
        }
        if (isDecimal) {
            switch (length) {
                case 4:
                    return buffer.getFloat();
                case 8:
                    return buffer.getDouble();
                default:
                    throw new IllegalArgumentException("Unsupported length: " + length);
            }
        }
        switch (length) {
            case 1:
                return isSigned ? (byte)buffer.get() : Byte.toUnsignedInt(buffer.get());
            case 2:
                return isSigned ? (short)buffer.getShort() : Short.toUnsignedInt(buffer.getShort());
            case 4:
                return isSigned ? (int)buffer.getInt() : Integer.toUnsignedLong(buffer.getInt());
            default:
                throw new IllegalArgumentException("Unsupported length: " + length);
        }
    }

    /**
     * Checks a given header for compatibility with the reader
     * @param IFD
     * @return empty string if compatible, error message otherwise
     */
    public static String isCompatible(IFDTag[] IFD) {
        boolean hasTileOffsets = false;
        int imageWidth = -1;
        int imageHeight = -1;
        int tileWidth = -1;
        int tileHeight = -1;
        for (IFDTag tag : IFD) {
            // Only 8 bit, 16 bit, 32 bit images are supported
            // This is common practice in TIFF readers
            // 12 bit values e.g. should instead be scaled to 16 bit
            switch (tag.getTagId()) {
                case BitsPerSample:
                    Number[] data = tag.getData();
                    for (int i = 0; i < data.length; i++) {
                        if (data[i].intValue() != 8 && data[i].intValue() != 16 && data[i].intValue() != 32) {
                            return "Unsupported bit depth: " + data[i];
                        }
                    }
                    break;
                case TileOffsets:
                    if (tag.getData().length > 0) {
                        hasTileOffsets = true;
                    }
                    break;
                case Compression:
                    // TODO: After implementing decompression, change this so compressed images are actually supported
                    if (tag.getData()[0].intValue() != 1 && tag.getData()[0].intValue() != 8) {
                        return "Unsupported compression: " + tag.getData()[0];
                    }
                    break;
                case ImageWidth:
                    imageWidth = tag.getData()[0].intValue();
                    break;
                case ImageLength:
                    imageHeight = tag.getData()[0].intValue();
                    break;
                case TileWidth:
                    tileWidth = tag.getData()[0].intValue();
                    break;
                case TileLength:
                    tileHeight = tag.getData()[0].intValue();
                    break;
            }
        }
        if (!hasTileOffsets) {
            return "No tile offsets found";
        }
        if (imageWidth % tileWidth != 0 || imageHeight % tileHeight != 0) {
            return "Image can't be split into tiles equally";
        }
        return "";
    }
}
