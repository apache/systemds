package com.ibm.bi.dml.sql.sqlcontrolprogram;

import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.utils.DMLRuntimeException;

public interface ISQLBlock extends SQLCodeGeneratable{
	public ProgramBlock getProgramBlock(Program p) throws DMLRuntimeException;
}
