package org.apache.sysds.test.functions.io.libsvm;

public class LIBSVMConfig {

    private String inSep;
    private String inIndSep;
    private int colCount;
    private String outSep;
    private String outIndSep;

    public LIBSVMConfig(String inSep, String inIndSep, int colCount, String outSep, String outIndSep) {
        this.inSep = inSep;
        this.inIndSep = inIndSep;
        this.colCount = colCount;
        this.outSep = outSep;
        this.outIndSep = outIndSep;
    }

    public String getInSep() {
        return inSep;
    }

    public void setInSep(String inSep) {
        this.inSep = inSep;
    }

    public String getInIndSep() {
        return inIndSep;
    }

    public void setInIndSep(String inIndSep) {
        this.inIndSep = inIndSep;
    }

    public int getColCount() {
        return colCount;
    }

    public void setColCount(int colCount) {
        this.colCount = colCount;
    }

    public String getOutSep() {
        return outSep;
    }

    public void setOutSep(String outSep) {
        this.outSep = outSep;
    }

    public String getOutIndSep() {
        return outIndSep;
    }

    public void setOutIndSep(String outIndSep) {
        this.outIndSep = outIndSep;
    }
}
