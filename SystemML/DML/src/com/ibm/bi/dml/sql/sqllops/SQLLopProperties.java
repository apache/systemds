/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.sql.sqllops;

public class SQLLopProperties 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum AGGREGATIONTYPE
	{
		SUM,
		MAX,
		MIN,
		NONE
	}
	
	public enum JOINTYPE
	{
		INNERJOIN,
		TWO_INNERJOINS,
		FULLOUTERJOIN,
		LEFTJOIN,
		RIGHTJOIN,
		CROSSJOIN,
		NONE
	}
	
	AGGREGATIONTYPE aggType = AGGREGATIONTYPE.NONE;
	JOINTYPE joinType = JOINTYPE.NONE;
	String opString = "Unknown";
	
	public AGGREGATIONTYPE getAggType() {
		return aggType;
	}
	public void setAggType(AGGREGATIONTYPE aggType) {
		this.aggType = aggType;
	}
	public JOINTYPE getJoinType() {
		return joinType;
	}
	public void setJoinType(JOINTYPE joinType) {
		this.joinType = joinType;
	}
	public String getOpString() {
		return opString;
	}
	public void setOpString(String opString) {
		this.opString = opString;
	}
	
}
