package com.ibm.bi.dml.sql.sqllops;

public class SQLLopProperties {
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
