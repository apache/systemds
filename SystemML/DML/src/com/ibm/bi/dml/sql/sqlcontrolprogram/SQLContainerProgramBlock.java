package com.ibm.bi.dml.sql.sqlcontrolprogram;

import java.util.ArrayList;

import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;


public class SQLContainerProgramBlock implements SQLBlockContainer{
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
		StringBuffer sb = new StringBuffer();
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
