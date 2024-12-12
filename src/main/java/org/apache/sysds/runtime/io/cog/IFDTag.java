package org.apache.sysds.runtime.io.cog;

public class IFDTag {
    private IFDTagDictionary tagId;
    // TODO: Implement enum for the data type (or now connect to the TIFFDataTypes class)
    // See TIFF specification for the different meanings
    private short dataType;
    private int dataCount;
    // TODO: Implement different data types
    private int[] data;

    public IFDTag(IFDTagDictionary tagId, short dataType, int dataCount, int[] data) {
        this.tagId = tagId;
        this.dataType = dataType;
        this.dataCount = dataCount;
        this.data = data;
    }

    public IFDTagDictionary getTagId() {
        return tagId;
    }

    public void setTagId(IFDTagDictionary tagId) {
        this.tagId = tagId;
    }

    public short getDataType() {
        return dataType;
    }

    public void setDataType(short dataType) {
        this.dataType = dataType;
    }

    public int getDataCount() {
        return dataCount;
    }

    public void setDataCount(int dataCount) {
        this.dataCount = dataCount;
    }

    public int[] getData() {
        return data;
    }

    public void setData(int[] data) {
        this.data = data;
    }
}
