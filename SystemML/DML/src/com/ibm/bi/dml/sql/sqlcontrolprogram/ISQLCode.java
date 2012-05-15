package com.ibm.bi.dml.sql.sqlcontrolprogram;

import com.ibm.bi.dml.runtime.instructions.Instruction;

public interface ISQLCode extends SQLCodeGeneratable {
	public Instruction[] getInstructions();
}
