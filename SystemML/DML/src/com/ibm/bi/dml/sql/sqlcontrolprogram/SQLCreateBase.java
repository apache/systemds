package com.ibm.bi.dml.sql.sqlcontrolprogram;

public abstract class SQLCreateBase implements ISQLCode {
	
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
