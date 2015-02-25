/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.sql.sqlcontrolprogram;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.sql.SQLDropTableInstruction;
import com.ibm.bi.dml.runtime.instructions.sql.SQLInstruction;
import com.ibm.bi.dml.sql.sqllops.SQLLops;


public class SQLOverwriteScalar extends SQLCreateTable 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	public String getMergedSelectStatement()
	{
		if(this.get_withs().size() > 0 && !trimmed)
		{
			SQLWithTable wt = this.get_withs().get(this.get_withs().size()-1);
			this.set_selectStatement(wt.get_selectStatement());
			this.get_withs().remove(wt);
			trimmed = true;
		}
		
		StringBuilder sb = new StringBuilder();
		if(this.get_withs().size() > 0)
			sb.append("   WITH ");
		
		for(int i = 0; i < this.get_withs().size(); i++)
		{
			if(i != 0)
				sb.append("   ");
			sb.append(this.get_withs().get(i).generateSQLString());
			if(i < this.get_withs().size() - 1)
				sb.append(",\r\n");
			else
				sb.append("\r\n");
		}
		sb.append(this.get_selectStatement());
		return sb.toString();
	}
	
	boolean trimmed = false;
	
	private String getCreateTableString(String tmp)
	{
		StringBuilder sb = new StringBuilder();
		sb.append("CREATE TABLE \"");
		sb.append(tmp);
		sb.append("\" (");
		sb.append(SQLLops.SCALARVALUECOLUMN);
		sb.append(" double precision ");
		
		sb.append(");\r\n");
		return sb.toString();
	}
	private String getInsertIntoString(String tmp)
	{
		StringBuilder sb = new StringBuilder();
	
		sb.append("INSERT INTO \"");
		sb.append(tmp);
		sb.append("\"\r\n(\r\n");
		
		sb.append(this.getMergedSelectStatement());
		
		sb.append(");\r\n");
		return sb.toString();
	}
	private String getDeleteTableString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append("call drop_if_exists('");
		sb.append(get_tableName());
		sb.append("');\r\n");
		return sb.toString();
	}
	private String getAlterTableString(String tmp)
	{
		StringBuilder sb = new StringBuilder();
		sb.append("ALTER TABLE \"");
		sb.append(tmp);
		sb.append("\" RENAME TO \"");
		sb.append(get_tableName());
		sb.append("\";\r\n");
		return sb.toString();
	}
	
	public String generateSQLString()
	{
		String tmp = "tmp_" + this.get_tableName();		
		return getCreateTableString(tmp) + getInsertIntoString(tmp) + getDeleteTableString() + getAlterTableString(tmp);
	}
	
	@Override
	public Instruction[] getInstructions() {
		String tmp = "tmp_" + this.get_tableName();
		Instruction[] instr = new Instruction[4];
		instr[0] = new SQLInstruction(getCreateTableString(tmp));
		instr[1] = new SQLInstruction(getInsertIntoString(tmp));
		instr[2] = new SQLDropTableInstruction(get_tableName());
		instr[3] = new SQLInstruction(getAlterTableString(tmp));
		return instr;
	}
	
	public int getStatementCount()
	{
		return 1;
	}
	
}
