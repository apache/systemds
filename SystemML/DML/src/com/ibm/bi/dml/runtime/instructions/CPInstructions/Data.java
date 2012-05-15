package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MetaData;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public abstract class Data {

	protected DataType dataType;
	protected ValueType valueType;
	
	protected Data(DataType dt, ValueType vt) {
		dataType = dt;
		valueType = vt;
	}
	
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
