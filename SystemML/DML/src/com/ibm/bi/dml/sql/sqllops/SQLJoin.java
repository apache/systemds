package com.ibm.bi.dml.sql.sqllops;

import java.util.ArrayList;

import com.ibm.bi.dml.sql.sqllops.SQLCondition.BOOLOP;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.JOINTYPE;


public class SQLJoin implements ISQLTableReference {
	
	public SQLJoin()
	{
		conditions = new ArrayList<SQLCondition>();
	}
	
	SQLLopProperties.JOINTYPE joinType;

	ISQLTableReference table1;
	ISQLTableReference table2;
	
	ArrayList<SQLCondition> conditions;
	
	public String getJoinTypeString()
	{
		if(joinType == JOINTYPE.FULLOUTERJOIN)
			return SQLLops.FULLOUTERJOIN;
		else if(joinType == JOINTYPE.LEFTJOIN)
			return SQLLops.LEFTJOIN;
		else if(joinType == JOINTYPE.CROSSJOIN)
			return ", ";
		else
			return SQLLops.JOIN;
	}
	
	public SQLLopProperties.JOINTYPE getJoinType() {
		return joinType;
	}
	public void setJoinType(SQLLopProperties.JOINTYPE joinType) {
		this.joinType = joinType;
	}
	
	public ISQLTableReference getTable1() {
		return table1;
	}

	public void setTable1(ISQLTableReference table1) {
		this.table1 = table1;
	}

	public ISQLTableReference getTable2() {
		return table2;
	}

	public void setTable2(ISQLTableReference table2) {
		this.table2 = table2;
	}

	public ArrayList<SQLCondition> getConditions() {
		return conditions;
	}
	
	public String toString()
	{
		StringBuffer sb = new StringBuffer();
		sb.append("( ");
		sb.append(table1.toString());
		
		sb.append(" " + getJoinTypeString() + " ");
		
		sb.append(table2.toString());
		
		if(getConditions().size() > 0)
			sb.append(" ON");
		for(SQLCondition co : this.getConditions())
		{
			if(co.boolOp == BOOLOP.AND)
				sb.append(" AND ");
			else if(co.boolOp == BOOLOP.OR)
				sb.append(" OR ");
			else
				sb.append(" ");
			sb.append(co.expression);
		}
		sb.append(" )");
		return sb.toString();
	}
}
