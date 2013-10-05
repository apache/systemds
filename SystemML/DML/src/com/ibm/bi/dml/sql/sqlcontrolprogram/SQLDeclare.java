/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.sql.sqlcontrolprogram;

import com.ibm.bi.dml.runtime.instructions.Instruction;

public class SQLDeclare implements ISQLCode 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	public SQLDeclare()
	{
		
	}
	
	public SQLDeclare(String varname, String type)
	{
		this._variable_name = varname;
		this._type = type;
	}
	
	String _variable_name;
	String _type;
	public String get_variable_name() {
		return _variable_name;
	}
	public void set_variable_name(String variableName) {
		_variable_name = variableName;
	}
	public String get_type() {
		return _type;
	}
	public void set_type(String type) {
		_type = type;
	}

	@Override
	public Instruction[] getInstructions() {
		return new Instruction[0];
	}

	@Override
	public String generateSQLString() {
		return get_variable_name() + " " + get_type() + ";\r\n";
	}

	@Override
	public int getStatementCount() {
		return 0;
	}
}
