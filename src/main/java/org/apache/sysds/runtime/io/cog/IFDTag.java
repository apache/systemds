package org.apache.sysds.runtime.io.cog;

/**
 * Represents a single tag in the IFD of a TIFF file
 */
public class IFDTag {
    private IFDTagDictionary tagId;
    private short dataType;
    private int dataCount;
    private Number[] data;

    public IFDTag(IFDTagDictionary tagId, short dataType, int dataCount, Number[] data) {
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

    public Number[] getData() {
        return data;
    }

    public void setData(Number[] data) {
        this.data = data;
    }
}
