/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.sql.sqlcontrolprogram;

import java.util.ArrayList;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.IfProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.sql.SQLScalarAssignInstruction;


public class SQLIfElseProgramBlock implements ISQLBlock 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
		StringBuilder sb = new StringBuilder();
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
	public ProgramBlock getProgramBlock(Program p) throws DMLRuntimeException {
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
