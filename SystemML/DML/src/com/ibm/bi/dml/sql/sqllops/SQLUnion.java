/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.sql.sqllops;

public class SQLUnion implements ISQLSelect 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum UNIONTYPE
	{
		NONE,
		UNIONALL,
		UNION
	}
	
	public SQLUnion()
	{
		
	}
	
	public SQLUnion(ISQLSelect s1, ISQLSelect s2, UNIONTYPE type)
	{
		select1 = s1;
		select2 = s2;
		unionType = type;
	}
	
	ISQLSelect select1;
	ISQLSelect select2;
	UNIONTYPE unionType;
	
	public ISQLSelect getSelect1() {
		return select1;
	}
	public void setSelect1(ISQLSelect select1) {
		this.select1 = select1;
	}
	public ISQLSelect getSelect2() {
		return select2;
	}
	public void setSelect2(ISQLSelect select2) {
		this.select2 = select2;
	}
	public UNIONTYPE getUnionType() {
		return unionType;
	}
	public void setUnionType(UNIONTYPE unionType) {
		this.unionType = unionType;
	}
	
	public String toString()
	{
		String union = NEWLINE + "UNION" + NEWLINE;
		if(this.unionType == UNIONTYPE.UNIONALL)
			union = NEWLINE + "UNION ALL" + NEWLINE;
		
		return select1.toString() + union + select2.toString();
	}
}
