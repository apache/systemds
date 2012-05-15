package com.ibm.bi.dml.sql.sqlcontrolprogram;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.SQLInstructions.SQLDropTableInstruction;
import com.ibm.bi.dml.runtime.instructions.SQLInstructions.SQLInstruction;
import com.ibm.bi.dml.sql.sqllops.SQLLops;


public class SQLOverwriteTable extends SQLCreateTable {
	
	public SQLOverwriteTable(boolean isTemp)
	{
		this.isTemp = isTemp;
	}
	
	public SQLOverwriteTable()
	{
	
	}
	
	boolean isTemp = false;
	
	private String getMergedSelectString()
	{
		if(this.get_withs().size() > 0 && !trimmed)
		{
			SQLWithTable wt = this.get_withs().get(this.get_withs().size()-1);
			this.set_selectStatement(wt.get_selectStatement());
			this.get_withs().remove(wt);
			trimmed = true;
		}
		StringBuffer sb = new StringBuffer();
		
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
		sb.append("    ");
		sb.append(this.get_selectStatement());
		
		return sb.toString();
	}
	
	private String getCreateTableString(String tmp)
	{
		StringBuffer sb = new StringBuffer();
		/*if(isTemp)
			sb.append("CREATE TEMP TABLE \"");
		else*/
		sb.append("CREATE TABLE \"");
		sb.append(tmp);
		sb.append("\" (");
		sb.append(SQLLops.ROWCOLUMN);
		sb.append(" INT8 NOT NULL, ");
		sb.append(SQLLops.COLUMNCOLUMN);
		sb.append(" INT8 NOT NULL, ");
		sb.append(SQLLops.VALUECOLUMN);
		sb.append(" double precision ) DISTRIBUTE ON (");
		sb.append(SQLLops.ROWCOLUMN);
		//sb.append(", ");
		//sb.append(SQLLops.COLUMNCOLUMN);
		sb.append(");\r\n");
		
		return sb.toString();
	}
	
	private String getInsertIntoString(String tmp)
	{
		StringBuffer sb = new StringBuffer();
		sb.append("INSERT INTO \"");
		sb.append(tmp);
		sb.append("\"\r\n(\r\n");
		
		//Assemble select clause
		sb.append(this.getMergedSelectString());
		sb.append("\r\n);\r\n");
		
		return sb.toString();
	}
	
	private String getDropTableString()
	{
		StringBuffer sb = new StringBuffer();
		sb.append("call drop_if_exists('");
		sb.append(get_tableName());
		sb.append("');\r\n");
		
		return sb.toString();
	}
	
	private String getAlterableString(String tmp)
	{
		StringBuffer sb = new StringBuffer();
		
		sb.append("ALTER TABLE \"");
		sb.append(tmp);
		sb.append("\" RENAME TO \"");
		sb.append(get_tableName());
		sb.append("\";\r\n");
		return sb.toString();
	}
	
	private String getGenerateStatisticsString()
	{
		return "GENERATE STATISTICS ON \"" + this.get_tableName() + "\";\r\n";
	}
	
	boolean trimmed = false;
	public String generateSQLString()
	{
		String tmp = null;
		tmp = "tmp_" + this.get_tableName();
		
		return getCreateTableString(tmp) + getInsertIntoString(tmp) + getDropTableString() + getAlterableString(tmp);
	}
	
	@Override
	public Instruction[] getInstructions() {
		String tmp = null;
		tmp = "tmp_" + this.get_tableName();
		
		Instruction[] instr = new Instruction[5];
		instr[0] = new SQLInstruction(getCreateTableString(tmp));
		instr[1] = new SQLInstruction(getInsertIntoString(tmp));
		instr[2] = new SQLDropTableInstruction(get_tableName());
		instr[3] = new SQLInstruction(getAlterableString(tmp));
		instr[4] = new SQLInstruction(getGenerateStatisticsString());
		return instr;
	}
	
	public int getStatementCount()
	{
		return 4;
	}
}
