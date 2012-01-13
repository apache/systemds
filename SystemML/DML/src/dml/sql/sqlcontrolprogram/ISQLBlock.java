package dml.sql.sqlcontrolprogram;

import dml.runtime.controlprogram.Program;
import dml.runtime.controlprogram.ProgramBlock;

public interface ISQLBlock extends SQLCodeGeneratable{
	public ProgramBlock getProgramBlock(Program p);
}
