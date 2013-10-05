/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.sql.sqlcontrolprogram;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.SQLInstructions.SQLInstruction;


public class SQLProcedureCall extends SQLCreateBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	String call;
	
	public SQLProcedureCall(String sql)
	{
		call = sql;
	}

	@Override
	public String generateSQLString() {
		return call;
	}

	@Override
	public Instruction[] getInstructions() {
		Instruction[] instr = new Instruction[1];
		instr[0] = new SQLInstruction(call);
		return instr;
	}

	@Override
	public int getStatementCount() {
		return 1;
	}
}
