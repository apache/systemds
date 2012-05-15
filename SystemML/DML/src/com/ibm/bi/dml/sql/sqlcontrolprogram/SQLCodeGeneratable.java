package com.ibm.bi.dml.sql.sqlcontrolprogram;

public interface SQLCodeGeneratable {
	String generateSQLString();
	public int getStatementCount();
}
