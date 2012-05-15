package com.ibm.bi.dml.sql.sqlcontrolprogram;

import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;

public interface ISQLBlock extends SQLCodeGeneratable{
	public ProgramBlock getProgramBlock(Program p);
}
