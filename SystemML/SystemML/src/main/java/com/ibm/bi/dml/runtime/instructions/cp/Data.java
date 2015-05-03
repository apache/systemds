/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import java.io.Serializable;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MetaData;


public abstract class Data implements Serializable 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = 9176228330268046168L;

	protected DataType dataType;
	protected ValueType valueType;
	
	public Data() {
		//default constructor for serialize
	}
	
	protected Data(DataType dt, ValueType vt) {
		dataType = dt;
		valueType = vt;
	}
	

	public abstract String getDebugName();
	
	public DataType getDataType() {
		return dataType;
	}

	public void setDataType(DataType dataType) {
		this.dataType = dataType;
	}

	public ValueType getValueType() {
		return valueType;
	}

	public void setValueType(ValueType valueType) {
		this.valueType = valueType;
	}

	public void setMetaData(MetaData md) throws DMLRuntimeException {
		throw new DMLRuntimeException("This method in the base class should never be invoked.");
	}
	
	public MetaData getMetaData() throws DMLRuntimeException {
		throw new DMLRuntimeException("This method in the base class should never be invoked.");
	}

	public void removeMetaData() throws DMLRuntimeException {
		throw new DMLRuntimeException("This method in the base class should never be invoked.");
	}

	public void updateMatrixCharacteristics(MatrixCharacteristics mc) throws DMLRuntimeException {
		throw new DMLRuntimeException("This method in the base class should never be invoked.");
	}
}
