/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.sql.sqlcontrolprogram;

public abstract class SQLCreateBase implements ISQLCode 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private String _selectStatement;
	private String _tableName;
	
	public String get_selectStatement() {
		return _selectStatement;
	}
	public void set_selectStatement(String selectStatement) {
		_selectStatement = selectStatement;
	}
	public String get_tableName() {
		return _tableName;
	}
	public void set_tableName(String tableName) {
		_tableName = tableName;
	}
	
	public abstract String generateSQLString();
}
