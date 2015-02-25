/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.sql.sqlcontrolprogram;

import com.ibm.bi.dml.runtime.instructions.Instruction;

public class SQLWithTable extends SQLCreateBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public String generateSQLString() {
		return String.format("\"%s\" AS ( %s )", this.get_tableName(), this.get_selectStatement());
	}

	@Override
	public Instruction[] getInstructions() {
		return new Instruction[0];
	}

	@Override
	public int getStatementCount() {
		return 0;
	}
	
}
