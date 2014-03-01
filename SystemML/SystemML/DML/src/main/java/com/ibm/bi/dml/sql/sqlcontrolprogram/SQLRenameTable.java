/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.sql.sqlcontrolprogram;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.SQLInstructions.SQLDropTableInstruction;
import com.ibm.bi.dml.runtime.instructions.SQLInstructions.SQLInstruction;


public class SQLRenameTable extends SQLCreateBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	public SQLRenameTable()
	{
		
	}
	
	public SQLRenameTable(String oldName, String newName)
	{
		this.set_tableName(oldName);
		_newName = newName;
	}
	
	String _newName;
	
	public String get_newName() {
		return _newName;
	}
	public void set_newName(String newName) {
		_newName = newName;
	}
 
	@Override
	public String generateSQLString() {
		String sql = 
			"call drop_if_exists('" + _newName + "');\r\n" +
			"ALTER TABLE \"" + this.get_tableName() + "\" RENAME TO \"" + _newName + "\";\r\n";
		return sql;
	}

	@Override
	public Instruction[] getInstructions() {
		Instruction[] instr = new Instruction[2];
		//instr[0] = new SQLInstruction("call drop_if_exists('" + _newName + "');\r\n");
		instr[0] = new SQLDropTableInstruction(_newName);
		instr[1] = new SQLInstruction("ALTER TABLE \"" + this.get_tableName() + "\" RENAME TO \"" + _newName + "\";\r\n");
		return instr;
	}

	@Override
	public int getStatementCount() {
		return 2;
	}
	
	
}
