/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
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
