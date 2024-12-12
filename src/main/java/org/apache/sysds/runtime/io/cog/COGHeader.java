package org.apache.sysds.runtime.io.cog;

import java.util.ArrayList;

public class COGHeader {
    private boolean isLittleEndian;
    private String GDALMetadata;
    private IFDTag[] IFD;
    // Do we even need this or will we throw it away?
    // If we keep this and write it again, we also need to write the additional images
    // So this will probably not make the cut
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
