/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.sql.sqlcontrolprogram;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.SQLInstructions.SQLDropTableInstruction;


public class SQLCleanup implements ISQLCode
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	public SQLCleanup()
	{
		
	}
	
	public SQLCleanup(boolean isView)
	{
		isview = isView;
	}
	
	boolean isview = false;
	
	public SQLCleanup(String tblname)
	{
		_tableName = tblname;
	}
	
	private String _tableName;

	public String get_tableName() {
		return _tableName;
	}

	public void set_tableName(String tableName) {
		_tableName = tableName;
	}
	
	public String generateSQLString()
	{
		return "call drop_if_exists('" + this.get_tableName() + "');\r\n";
		
		/*if(isview)
			return "DROP VIEW \"" + this.get_tableName() + "\";\r\n";
		else
			return "DROP TABLE \"" + this.get_tableName() + "\";\r\n";*/
	}

	@Override
	public Instruction[] getInstructions() {
		Instruction[] instr = new Instruction[1];
		instr[0] = new SQLDropTableInstruction(get_tableName());
		//instr[0] = new SQLInstruction("call drop_if_exists('" + this.get_tableName() + "');");
		return instr;
	}

	@Override
	public int getStatementCount() {
		return 1;
	}
}
