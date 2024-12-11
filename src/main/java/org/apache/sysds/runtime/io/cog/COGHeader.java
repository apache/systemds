package org.apache.sysds.runtime.io.cog;

public class COGHeader {
    private boolean isLittleEndian;
    private String GDALMetadata;
    private IFDTag[] IFD;

    public COGHeader(boolean isLittleEndian) {
        this.isLittleEndian = isLittleEndian;
        GDALMetadata = "";
    }

    public void setIFD(IFDTag[] IFD) {
        this.IFD = IFD;
    }

    public IFDTag[] getIFD() {
        return IFD;
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

    public int parseBytes(byte[] bytes) {
        return parseBytes(bytes, bytes.length, 0);
    }

    public int parseBytes(byte[] bytes, int length) {
        return parseBytes(bytes, length, 0);
    }

    public int parseBytes(byte[] bytes, int length, int offset) {
        int sum = 0;
        if (isLittleEndian) {
            for (int i = 0; i < length; i++) {
                sum |= (bytes[offset + i] & 0xFF) << (8 * i);
            }
        } else {
            for (int i = 0; i < length; i++) {
                sum |= (bytes[offset + i] & 0xFF) << (8 * (length - i - 1));
            }
        }
        return sum;
    }
}
