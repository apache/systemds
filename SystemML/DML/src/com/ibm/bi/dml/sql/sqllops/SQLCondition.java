/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.sql.sqllops;

public class SQLCondition 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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