/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.parser.Expression.*;
import com.ibm.bi.dml.runtime.instructions.Instruction;


public class CPOperand 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private String _name;
	private ValueType _valueType;
	private DataType _dataType;
	private boolean _isLiteral;
	
	public CPOperand(String str) {
		split(str);
		//split_by_value_type_prefix(str);
	}
	
	public CPOperand() {
		_name = "";
		_valueType = ValueType.UNKNOWN;
		_dataType = DataType.UNKNOWN;
		_isLiteral = false;
	}
	
	public CPOperand(String name, ValueType vt, DataType dt ) {
		_name = name;
		_valueType = vt;
		_dataType = dt;
		_isLiteral = false;
	}
	
	public CPOperand(String name, ValueType vt, DataType dt, boolean literal ) {
		_name = name;
		_valueType = vt;
		_dataType = dt;
		_isLiteral = literal;
	}
	
	public String get_name() {
		return _name;
	}
	
	public ValueType get_valueType() {
		return _valueType;
	}
	
	public DataType get_dataType() {
		return _dataType;
	}
	
	public boolean isLiteral() {
		return _isLiteral;
	}
	
	public void set_name(String name) {
		_name = name;
	}
	
	public void set_valueType(ValueType vt) {
		_valueType = vt;
	}
	
	public void set_dataType(DataType dt) {
		_dataType = dt;
	}
	
	public void set_literal(boolean literal) {
		_isLiteral = literal;
	}
	
	public void split_by_value_type_prefix ( String str ) {
		String[] opr = str.split(Lop.VALUETYPE_PREFIX);
		_name = opr[0];
		_valueType = ValueType.valueOf(opr[1]);
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
		else {
			_name = opr[0];
			_valueType = ValueType.valueOf(opr[1]);
		}
	}
	
	public void copy(CPOperand o){
		_name = o.get_name();
		_valueType = o.get_valueType();
		_dataType = o.get_dataType();
	}
}
