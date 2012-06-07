package com.ibm.bi.dml.sql.sqlcontrolprogram;

import java.util.ArrayList;

import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.SQLInstructions.SQLCommitInstruction;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class SQLProgramBlock implements ISQLBlock {
	
	public SQLProgramBlock()
	{
		_creates = new ArrayList<SQLCreateBase>();
		_writes = new ArrayList<SQLCreateBase>();
		_renames = new ArrayList<SQLRenameTable>();
		_cleanups = new ArrayList<SQLCleanup>();
	}
	
	private ArrayList<SQLCreateBase> _creates;
	private ArrayList<SQLCreateBase> _writes;
	private ArrayList<SQLRenameTable> _renames;
	private ArrayList<SQLCleanup> _cleanups;
	
	/**
	 * Returns a list with statements that might other variables and therefore must be executed at the end
	 * @return
	 */
	public ArrayList<SQLCreateBase> get_writes() {
		return _writes;
	}

	/**
	 * Returns a list with statements for calculating values
	 * @return
	 */
	public ArrayList<SQLCreateBase> get_creates() {
		return _creates;
	}
	
	public ArrayList<SQLRenameTable> get_renames() {
		return _renames;
	}

	/**
	 * Returns a list with cleanup statements
	 * @return
	 */
	public ArrayList<SQLCleanup> get_cleanups() {
		return _cleanups;
	}
	
	@Override
	public String generateSQLString()
	{
		StringBuffer sb = new StringBuffer();
		for(SQLCreateBase ct : get_creates())
		{
			sb.append(ct.generateSQLString());
		}
		sb.append("-------Writing--------\r\n");
		for(SQLCreateBase ct : get_writes())
		{
			sb.append(ct.generateSQLString());
		}
		if(this._renames.size() > 0)
		{
			sb.append("-------Renaming--------\r\n");
			for(SQLRenameTable rt : get_renames())
			{
				sb.append(rt.generateSQLString());
			}
		}
		if(get_cleanups().size() > 0)
		{
			sb.append("-- cleanup tables\r\n");
			for(SQLCleanup cu : get_cleanups())
				sb.append(cu.generateSQLString());
		}
		return sb.toString();
	}
	
	public int getStatementCount()
	{
		return this.get_creates().size() + this.get_writes().size();
	}

	@Override
	public ProgramBlock getProgramBlock(Program p) throws DMLRuntimeException {
		ProgramBlock block = new ProgramBlock(p);
		for(ISQLCode c : this._creates)
		{
			for(Instruction i : c.getInstructions())
				block.addInstruction(i);
		}
		for(ISQLCode c : this._writes)
		{
			for(Instruction i : c.getInstructions())
				block.addInstruction(i);
		}
		for(ISQLCode c : this._renames)
		{
			for(Instruction i : c.getInstructions())
				block.addInstruction(i);
		}
		for(ISQLCode c : this._cleanups)
		{
			for(Instruction i : c.getInstructions())
				block.addInstruction(i);
		}
		block.addInstruction(new SQLCommitInstruction());
		return block;
	}
}