/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.sql.sqlcontrolprogram;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.sql.SQLPrintInstruction;


public class SQLPrint extends SQLCreateTable
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public String generateSQLString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.get_tableName());
		sb.append(" := ");
		sb.append("( ");
		if(this.get_withs().size() > 0)
			sb.append(" WITH ");
		for(int i= 0; i < this.get_withs().size(); i++)
		{
			SQLWithTable wt = this.get_withs().get(i);
			sb.append(wt.generateSQLString());
			if(i < this.get_withs().size() - 1)
				sb.append(",\r\n");
			else
				sb.append("\r\n");
		}
		sb.append(this.get_selectStatement());
		sb.append(");\r\n");

		sb.append("RAISE NOTICE 'Message: %', ");
		sb.append(this.get_tableName());
		sb.append(";\r\n");
		return sb.toString();
	}
	
	@Override
	public Instruction[] getInstructions() {
		Instruction[] instr = new Instruction[1];
		instr[0] = new SQLPrintInstruction(this.get_selectStatement());
		return instr;
	}

	@Override
	public int getStatementCount() {
		return 1;
	}
}
