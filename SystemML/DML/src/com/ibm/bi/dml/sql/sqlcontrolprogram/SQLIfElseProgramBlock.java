package com.ibm.bi.dml.sql.sqlcontrolprogram;

import java.util.ArrayList;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.IfProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.SQLInstructions.SQLScalarAssignInstruction;


public class SQLIfElseProgramBlock implements ISQLBlock {

	SQLContainerProgramBlock _ifbody;
	SQLContainerProgramBlock _elsebody;
	String predicateTableName;
	String _predicate;
	
	
	public SQLContainerProgramBlock get_ifbody() {
		return _ifbody;
	}

	public void set_ifbody(SQLContainerProgramBlock ifbody) {
		_ifbody = ifbody;
	}

	public SQLContainerProgramBlock get_elsebody() {
		return _elsebody;
	}

	public void set_elsebody(SQLContainerProgramBlock elsebody) {
		_elsebody = elsebody;
	}
	
	public String get_predicate() {
		return _predicate;
	}

	public void set_predicate(String predicate) {
		_predicate = predicate;
	}
	
	
	public String getPredicateTableName() {
		return predicateTableName;
	}

	public void setPredicateTableName(String predicateTableName) {
		this.predicateTableName = predicateTableName;
	}

	@Override
	public String generateSQLString() {
		StringBuffer sb = new StringBuffer();
		sb.append("IF ( ");
		sb.append(get_predicate());
		sb.append(" ) THEN\r\n");
		sb.append(this.get_ifbody().generateSQLString());
		if(this.get_elsebody() != null)
		{
			sb.append("ELSE\r\n");
			sb.append(this.get_elsebody().generateSQLString());
		}
		sb.append("END IF;\r\n");
		return sb.toString();
	}

	@Override
	public int getStatementCount() {
		int total = 0;
		if(this._elsebody != null)
			for(SQLCodeGeneratable g : this._elsebody._blocks)
				total += g.getStatementCount();
		for(SQLCodeGeneratable g : this._ifbody._blocks)
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
		
		IfProgramBlock block = new IfProgramBlock(p, pred);
		
		for(ISQLBlock b : _ifbody.get_blocks())
		{
			block.getChildBlocksIfBody().add(b.getProgramBlock(p));
		}
		
		if(_elsebody != null)
		for(ISQLBlock b : _elsebody.get_blocks())
		{
			block.getChildBlocksElseBody().add(b.getProgramBlock(p));
		}
		
		return block;
	}
	
	
}
