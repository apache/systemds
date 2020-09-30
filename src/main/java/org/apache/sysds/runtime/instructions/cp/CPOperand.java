/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.instructions.cp;

import org.apache.commons.lang.builder.ToStringBuilder;
import org.apache.commons.lang.builder.ToStringStyle;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;

public class CPOperand
{
	private String _name;
	private ValueType _valueType;
	private DataType _dataType;
	private boolean _isLiteral;
	private ScalarObject _literal;
	private PrivacyConstraint _privacyConstraint;
	
	public CPOperand() {
		this("", ValueType.UNKNOWN, DataType.UNKNOWN);
	}

	public CPOperand(String str) {
		this("", ValueType.UNKNOWN, DataType.UNKNOWN);
		split(str);
	}
	
	public CPOperand(String name, ValueType vt, DataType dt) {
		this(name, vt, dt, false);
	}

	public CPOperand(String name, ValueType vt, DataType dt, boolean literal) {
		_name = name;
		_valueType = vt;
		_dataType = dt;
		_isLiteral = literal;
	}

	public CPOperand(ScalarObject so) {
		_name = so.getStringValue();
		_valueType = so.getValueType();
		_dataType = DataType.SCALAR;
		_isLiteral = true;
	}

	public CPOperand(CPOperand variable){
		_name = variable._name;
		_valueType = variable._valueType;
		_dataType = variable._dataType;
		_isLiteral = variable._isLiteral;
		_literal = variable._literal;
	}
	
	public CPOperand(String name, Data dat) {
		_name = name;
		_valueType = dat.getValueType();
		_dataType = dat.getDataType();
		_isLiteral = false;
	}

	public String getName() {
		return _name;
	}
	
	public ValueType getValueType() {
		return _valueType;
	}
	
	public DataType getDataType() {
		return _dataType;
	}
	
	public boolean isMatrix() {
		return _dataType.isMatrix();
	}
	
	public boolean isFrame() {
		return _dataType.isFrame();
	}

	public boolean isTensor() {
		return _dataType.isTensor();
	}

	public boolean isList() {
		return _dataType.isList();
	}

	public boolean isScalar() {
		return _dataType.isScalar();
	}
	
	public boolean isLiteral() {
		return _isLiteral;
	}
	
	public ScalarObject getLiteral() {
		if( !_isLiteral )
			throw new DMLRuntimeException("CPOperand is not a literal.");
		if( _literal == null )
			_literal = ScalarObjectFactory.createScalarObject(_valueType, _name);
		return _literal;
	}
	
	public void setName(String name) {
		_name = name;
		_literal = null;
	}
	
	public void setLiteral(ScalarObject literal) {
		_name = String.valueOf(literal);
		_literal = literal;
		_isLiteral = (_literal!=null);
	}

	public void split(String str){
		String[] opr = str.split(Instruction.VALUETYPE_PREFIX);
		if ( opr.length == 4 ) {
			_name = opr[0];
			_dataType = DataType.valueOf(opr[1]);
			_valueType = ValueType.valueOf(opr[2]);
			_isLiteral = Boolean.parseBoolean(opr[3]);
		}
		else if ( opr.length == 3 ) {
			_name = opr[0];
			_dataType = DataType.valueOf(opr[1]);
			_valueType = ValueType.valueOf(opr[2]);
			_isLiteral = false;
		}
		else if ( opr.length == 1 ) {
			//note: for literals in MR instructions
			_name = opr[0];
			_dataType = DataType.SCALAR;
			_valueType = ValueType.FP64;
			_isLiteral = true;
		}
		else {
			_name = opr[0];
			_valueType = ValueType.valueOf(opr[1]);
		}
	}

	public PrivacyConstraint getPrivacyConstraint() {
		return _privacyConstraint;
	}

	public void setPrivacyConstraint(PrivacyConstraint privacyConstraint) {
		_privacyConstraint = privacyConstraint;
	}

	@Override
	public String toString() {
		return ToStringBuilder.reflectionToString(this, ToStringStyle.SHORT_PREFIX_STYLE);
	}

	public String getLineageLiteral() {
		return InstructionUtils.concatOperandParts(
			getName(), getDataType().name(),
			getValueType().name(), String.valueOf(isLiteral()));
	}
	
	public String getLineageLiteral(ScalarObject so) {
		return InstructionUtils.concatOperandParts(
			so.toString(), getDataType().name(),
			getValueType().name(), String.valueOf(isLiteral()));
	}
}
