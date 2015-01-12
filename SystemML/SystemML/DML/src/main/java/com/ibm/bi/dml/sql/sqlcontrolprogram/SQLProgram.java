/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.sql.sqlcontrolprogram;

import java.util.ArrayList;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.sql.SQLCommitInstruction;


public class SQLProgram implements SQLBlockContainer
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	public SQLProgram()
	{
		_blocks = new ArrayList<ISQLBlock>();
		_variableDeclarations = new ArrayList<SQLDeclare>();
		_cleanups = new ArrayList<SQLCleanup>();
	}
	
	private ArrayList<SQLDeclare> _variableDeclarations;
	private ArrayList<SQLCleanup> _cleanups;
	private String _name = "noname";
	private ArrayList<ISQLBlock> _blocks;

	private static final String PROCDECLARATION = "CREATE OR REPLACE PROCEDURE %s() RETURNS INT4\r\nLANGUAGE NZPLSQL AS\r\n";
	
	public ArrayList<SQLCleanup> get_cleanups() {
		return _cleanups;
	}

	public String get_name() {
		return _name;
	}

	public void set_name(String name) {
		_name = name;
	}

	public ArrayList<ISQLBlock> get_blocks() {
		return _blocks;
	}
	
	public ArrayList<SQLDeclare> get_variableDeclarations() {
		return _variableDeclarations;
	}
	
	@Override
	public String generateSQLString()
	{
		StringBuffer sb = new StringBuffer();
		sb.append(String.format(PROCDECLARATION, this.get_name()));
		sb.append("BEGIN_PROC\r\n");
		
		if(this._variableDeclarations.size() > 0)
		{
			sb.append("DECLARE\r\n");
			for(SQLDeclare dec : _variableDeclarations)
				sb.append(dec.generateSQLString());
		}
		sb.append("BEGIN\r\n");
		
		for(SQLCodeGeneratable block : get_blocks())
		{
			sb.append("------------------- Block start -------------------\r\n");
			sb.append(block.generateSQLString());
		}
		
		if(get_cleanups().size() > 0)
		{
			sb.append("-- cleanup tables\r\n");
			for(SQLCleanup cu : get_cleanups())
				sb.append(cu.generateSQLString());
		}
		sb.append("END;\r\n");
		sb.append("END_PROC;\r\n");
		
		
		return removePlaceholders(sb.toString());
	}
	
	private String removePlaceholders(String s)
	{
		return s.replace("##", "");
	}
	
	public boolean cleansUp(String tableName)
	{
		for(SQLCleanup cu : this.get_cleanups())
			if(cu.get_tableName().equals(tableName))
				return true;
		return false;
	}

	@Override
	public int getStatementCount() {
		int total = 0;
		for(SQLCodeGeneratable g : this._blocks)
			total += g.getStatementCount();
		return total;
	}

	@Override
	public ProgramBlock getProgramBlock(Program p) {
		return null;
	}
	
	public Program getProgram() throws DMLRuntimeException
	{
		Program p = new Program();
		for(ISQLBlock b : this._blocks)
		{
			p.addProgramBlock(b.getProgramBlock(p));
		}
		ProgramBlock cleanups = new ProgramBlock(p);
		for(ISQLCode c : this._cleanups)
		{
			for(Instruction i : c.getInstructions())
				cleanups.addInstruction(i);
		}
		//Last commit for cleanups
		cleanups.addInstruction(new SQLCommitInstruction());
		p.addProgramBlock(cleanups);
		return p;
	}
}
