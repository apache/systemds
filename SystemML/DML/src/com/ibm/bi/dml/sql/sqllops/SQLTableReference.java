/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.sql.sqllops;

public class SQLTableReference implements ISQLTableReference 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	public SQLTableReference()
	{
		
	}
	public SQLTableReference(String name, String alias)
	{
		this.name = name;
		this.alias = alias;
	}
	public SQLTableReference(String name)
	{
		this.name = name;
	}
	
	String name;
	String alias;
	
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public String getAlias() {
		return alias;
	}
	public void setAlias(String alias) {
		this.alias = alias;
	}
	
	public String toString()
	{
		String s = this.name;
		if(this.alias != null && this.alias.length() > 0)
			s += " " + alias;
		return s;
	}
}
