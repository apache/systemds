package com.ibm.bi.dml.runtime.matrix.mapred;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;

public class IndexedMatrixValue extends CachedMapElement {
	private MatrixIndexes indexes=new MatrixIndexes();
	private MatrixValue value;
	private Class<? extends MatrixValue> valueClass=MatrixBlock.class;
	public IndexedMatrixValue(Class<? extends MatrixValue> cls)
	{
		valueClass=cls;
		try {
			value=valueClass.newInstance();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
	public IndexedMatrixValue(MatrixIndexes ind, MatrixValue b)
	{
		indexes.setIndexes(ind);
		valueClass=b.getClass();
		try {
			value=valueClass.newInstance();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		value.copy(b);
	}
	public IndexedMatrixValue(IndexedMatrixValue that)
	{
		set(that.indexes, that.value);
	}
	@Override
	public CachedMapElement duplicate() {
		return new IndexedMatrixValue(this.indexes, this.value);
	}
	@Override
	public void set(CachedMapElement elem) {
		if(elem instanceof IndexedMatrixValue)
		{
			IndexedMatrixValue that=(IndexedMatrixValue) elem;
			set(that.indexes, that.value);
		}
	}
	public MatrixIndexes getIndexes()
	{
		return indexes;
	}
	public MatrixValue getValue()
	{
		return value;
	}
	public void set(MatrixIndexes indexes2, MatrixValue block2) {
		this.indexes.setIndexes(indexes2);
		this.value.copy(block2);
	}
	public String toString()
	{
		return "("+indexes.getRowIndex()+", "+indexes.getColumnIndex()+"): \n"+value;
	}
}
