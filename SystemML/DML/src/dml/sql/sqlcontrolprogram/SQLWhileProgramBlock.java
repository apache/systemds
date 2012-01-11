package dml.sql.sqlcontrolprogram;

import java.util.ArrayList;

import dml.parser.Expression.ValueType;
import dml.runtime.controlprogram.Program;
import dml.runtime.controlprogram.ProgramBlock;
import dml.runtime.controlprogram.WhileProgramBlock;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.SQLInstructions.SQLScalarAssignInstruction;

public class SQLWhileProgramBlock implements SQLBlockContainer {

	public SQLWhileProgramBlock()
	{
		_blocks = new ArrayList<ISQLBlock>();
	}
	
	String _predicate;
	String predicateTableName;
	ArrayList<ISQLBlock> _blocks;
	
	public ArrayList<ISQLBlock> get_blocks() {
		return _blocks;
	}

	public String getPredicateTableName() {
		return predicateTableName;
	}

	public void setPredicateTableName(String predicateTableName) {
		this.predicateTableName = predicateTableName;
	}
	
	public String get_predicate() {
		return _predicate;
	}

	public void set_predicate(String predicate) {
		_predicate = predicate;
	}
	
	public String generateSQLString()
	{
		StringBuffer sb = new StringBuffer();
		sb.append("WHILE ( ");
		sb.append(get_predicate());
		sb.append(" ) LOOP\r\n");
		
		for(SQLCodeGeneratable b : this.get_blocks())
		{
			sb.append("------------------- Block start -------------------\r\n");
			sb.append(b.generateSQLString());
		}
		sb.append("END LOOP;\r\n");
		return sb.toString();
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
		String s = this.getPredicateTableName();
		if(s.startsWith("##"))
			s = s.substring(2, this.getPredicateTableName().length()-2);
		
		SQLScalarAssignInstruction a = new SQLScalarAssignInstruction(
				s, this.get_predicate(), ValueType.BOOLEAN);
		ArrayList<Instruction> pred = new ArrayList<Instruction>();
		pred.add(a);
		
		WhileProgramBlock block = new WhileProgramBlock(p, pred);
		
		for(ISQLBlock b : this._blocks)
			block.addProgramBlock(b.getProgramBlock(p));
		return block;
	}
}
