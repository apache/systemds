/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.sql.sqlcontrolprogram;

import java.util.ArrayList;

import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;


public class SQLContainerProgramBlock implements SQLBlockContainer
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public SQLContainerProgramBlock()
	{
		_blocks = new ArrayList<ISQLBlock>();
	}
	
	ArrayList<ISQLBlock> _blocks;
	
	public ArrayList<ISQLBlock> get_blocks() {
		return _blocks;
	}

	public String generateSQLString()
	{
		StringBuilder sb = new StringBuilder();
		for(ISQLBlock b : this.get_blocks())
		{
			sb.append("------------------- Block start -------------------\r\n");
			sb.append(b.generateSQLString());
		}
		return sb.toString();
	}
	
	@Override
	public int getStatementCount() {
		int total = 1;//One for the predicate
		for(ISQLBlock g : this._blocks)
			total += g.getStatementCount();
		return total;
	}

	@Override
	public ProgramBlock getProgramBlock(Program p) {
		return null;
	}
}
