package com.ibm.bi.dml.runtime.matrix.io;

import org.apache.hadoop.io.Writable;

public interface Converter<K1 extends Writable, V1 extends Writable, 
K2 extends Writable, V2 extends Writable> {
	public void convert(K1 k1, V1 v1);
	public void setBlockSize(int rl, int cl);
	public boolean hasNext();
	public Pair<K2, V2> next();
}
