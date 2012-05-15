package com.ibm.bi.dml.sql.sqllops;

public class SQLCondition {
	public enum BOOLOP
	{
		NONE,
		AND,
		OR
	}
	
	public SQLCondition(BOOLOP op, String exp)
	{
		boolOp = op;
		expression = exp;
	}
	
	public SQLCondition(String exp)
	{
		expression = exp;
	}
	
	BOOLOP boolOp = BOOLOP.NONE;
	String expression;
	
	public BOOLOP getBoolOp() {
		return boolOp;
	}
	public void setBoolOp(BOOLOP boolOp) {
		this.boolOp = boolOp;
	}
	public String getExpression() {
		return expression;
	}
	public void setExpression(String expression) {
		this.expression = expression;
	}
	
	public static String boolOp2String(BOOLOP op)
	{
		if(op == BOOLOP.AND)
			return "AND";
		else if(op == BOOLOP.OR)
			return "OR";
		return "";
	}
}