package org.apache.sysds.runtime.data;

public abstract class DenseBlockBool extends DenseBlockDRB{

    protected DenseBlockBool(int[] dims) {
        super(dims);
    }
    public abstract DenseBlock set(int r, int c, boolean v);

    public abstract boolean getBoolean(int r, int c);
}
