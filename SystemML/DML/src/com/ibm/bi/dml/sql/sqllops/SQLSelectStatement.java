package com.ibm.bi.dml.sql.sqllops;

import java.util.ArrayList;

import com.ibm.bi.dml.sql.sqllops.SQLCondition.BOOLOP;


public class SQLSelectStatement implements ISQLSelect {
	
	public SQLSelectStatement()
	{
		columns = new ArrayList<String>();
		wheres = new ArrayList<SQLCondition>();
		groupBys = new ArrayList<String>();
		havings = new ArrayList<SQLCondition>();
	}
	
	boolean hasFromPart = true;
	ArrayList<String> columns;
	ISQLTableReference table;
	ArrayList<SQLCondition> wheres;
	ArrayList<String> groupBys;
	ArrayList<SQLCondition> havings;
	
	
	
	public ISQLTableReference getTable() {
		return table;
	}
	public void setTable(ISQLTableReference table) {
		this.table = table;
	}
	public boolean hasFromPart() {
		return hasFromPart;
	}
	public void setHasFromPart(boolean hasFromPart) {
		this.hasFromPart = hasFromPart;
	}
	public ArrayList<String> getColumns() {
		return columns;
	}
	public ArrayList<SQLCondition> getWheres() {
		return wheres;
	}
	public ArrayList<String> getGroupBys() {
		return groupBys;
	}
	public ArrayList<SQLCondition> getHavings() {
		return havings;
	}
	
	public String toString()
	{
		StringBuffer sb = new StringBuffer();
		sb.append("SELECT ");
		//Selected Columns
		for(int i = 0; i < columns.size(); i++)
		{
			if(i != 0)
				sb.append(", ");
			sb.append(columns.get(i));
		}
		
		if(table != null)
			sb.append(" FROM " + table.toString());
		
		//Filter using where clauses
		if(wheres.size() > 0)
		{
			sb.append(NEWLINE + "WHERE");
			for(SQLCondition co : wheres)
			{
				if(co.getBoolOp() != BOOLOP.NONE)
					sb.append(" " + SQLCondition.boolOp2String(co.boolOp));
				sb.append(" " + co.getExpression());
			}
		}
		//Group by clauses
		if(groupBys.size() > 0)
		{
			sb.append(NEWLINE + "GROUP BY");
			for(int i = 0; i < groupBys.size(); i++)
			{
				if(i > 0)
					sb.append(",");
				sb.append(" " + groupBys.get(i));
			}
		}
		//Having clauses
		if(havings.size() > 0)
		{
			sb.append(NEWLINE + "HAVING");
			for(SQLCondition co : havings)
			{
				if(co.getBoolOp() != BOOLOP.NONE)
					sb.append(" " + SQLCondition.boolOp2String(co.boolOp));
				sb.append(" " + co.getExpression());
			}
		}
		return sb.toString();
	}
}
