/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;

public class IndexedMatrixValue extends CachedMapElement 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private MatrixIndexes indexes=new MatrixIndexes();
	private MatrixValue value=null;
	private Class<? extends MatrixValue> valueClass=MatrixBlock.class;
	
	public IndexedMatrixValue()
	{
	}
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
	
	public void shallowSetValue(MatrixValue v2)
	{
		value=v2;
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
