/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.Serializable;

import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;

public class IndexedMatrixValue implements Serializable
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = 6723389820806752110L;

	private MatrixIndexes _indexes = null;
	private MatrixValue   _value = null;
	
	public IndexedMatrixValue()
	{
		_indexes = new MatrixIndexes();
	}
	
	public IndexedMatrixValue(Class<? extends MatrixValue> cls)
	{
		this();
		
		//create new value object for given class
		try {
			_value=cls.newInstance();
		} 
		catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
	
	public IndexedMatrixValue(MatrixIndexes ind, MatrixValue b)
	{
		this();
		
		_indexes.setIndexes(ind);
		_value = b;
	}

	public IndexedMatrixValue(IndexedMatrixValue that)
	{
		this(that._indexes, that._value); 
	}

	
	public MatrixIndexes getIndexes()
	{
		return _indexes;
	}
	
	public MatrixValue getValue()
	{
		return _value;
	}
	
	public void set(MatrixIndexes indexes2, MatrixValue block2) {
		_indexes.setIndexes(indexes2);
		_value = block2;
	}
	
	public String toString()
	{
		return "("+_indexes.getRowIndex()+", "+_indexes.getColumnIndex()+"): \n"+_value;
	}
}
