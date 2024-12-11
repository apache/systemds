package org.apache.sysds.runtime.io.cog;

public class IFDTag {
    private IFDTagDictionary tagId;
    // TODO: Implement enum for the data type
    // See TIFF specification for the different meanings
    private short dataType;
    private int dataCount;
    private int dataOrOffset;

    public IFDTag(IFDTagDictionary tagId, short dataType, int dataCount, int dataOrOffset) {
        this.tagId = tagId;
        this.dataType = dataType;
        this.dataCount = dataCount;
        this.dataOrOffset = dataOrOffset;
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

    public int getDataOrOffset() {
        return dataOrOffset;
    }

    public void setDataOrOffset(int dataOrOffset) {
        this.dataOrOffset = dataOrOffset;
    }
}
