package dml.utils.visualize;

import java.util.ArrayList;
import java.util.Iterator;

import dml.hops.Hops;
import dml.hops.Hops.VISIT_STATUS;
import dml.parser.DMLProgram;
import dml.parser.ForStatement;
import dml.parser.ForStatementBlock;
import dml.parser.FunctionStatement;
import dml.parser.IfStatement;
import dml.parser.IfStatementBlock;
import dml.parser.StatementBlock;
import dml.parser.WhileStatement;
import dml.parser.WhileStatementBlock;
import dml.sql.sqllops.SQLLops;
import dml.sql.sqllops.SQLLopProperties.JOINTYPE;
import dml.utils.HopsException;
import dml.utils.LanguageException;

public class SQLLopGraph {
	public static String getSQLLopGraphString(DMLProgram dmlp, String title, int x, int y, String basePath) throws HopsException, LanguageException
	{
		String graphString = new String("digraph G { \n node [ label = \"\\N\", style=filled ]; edge [dir=back]; \n ");
		
		for (String namespaceKey : dmlp.getNamespaces().keySet()){
			for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet()){
				FunctionStatement fstmt = (FunctionStatement)dmlp.getFunctionStatementBlock(namespaceKey, fname).getStatement(0);
				for (StatementBlock sb : fstmt.getBody()){
					graphString += getSQLLopGraphString(sb);
				}
			}
		}
		
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			graphString += getSQLLopGraphString(current);
		}
		
		graphString += "\n } \n";
		return graphString;
	}
	
	public static String getSQLLopGraphString(StatementBlock current) throws HopsException
	{
		String graphString = "";
		
		if (current instanceof WhileStatementBlock) {
			// Handle Predicate
			SQLLops predicateSQLLop = ((WhileStatementBlock) current).getPredicateHops().get_sqllops();
			String predicateString = prepareSQLLopNodeList(predicateSQLLop);
			graphString += predicateString;
			
			// handle children
			WhileStatement wstmt = (WhileStatement)((WhileStatementBlock)current).getStatement(0);
			for (StatementBlock sb : wstmt.getBody()){	
				graphString += getSQLLopGraphString(sb);
			}
			
		}
		else if (current instanceof ForStatementBlock) {
			// Handle Predicate
			SQLLops predicateSQLLop = ((ForStatementBlock) current).getPredicateHops().get_sqllops();
			String predicateString = prepareSQLLopNodeList(predicateSQLLop);
			graphString += predicateString;
			
			// handle children
			ForStatement fstmt = (ForStatement)((ForStatementBlock)current).getStatement(0);
			for (StatementBlock sb : fstmt.getBody()){	
				graphString += getSQLLopGraphString(sb);
			}
			
		}
		
		else if (current instanceof IfStatementBlock) {
			// Handle Predicate
			SQLLops predicateSQLLop = ((IfStatementBlock) current).getPredicateHops().get_sqllops();
			String predicateString = prepareSQLLopNodeList(predicateSQLLop);
			graphString += predicateString;
			
			// handle children
			IfStatement ifstmt = (IfStatement)((IfStatementBlock)current).getStatement(0);
			for (StatementBlock sb : ifstmt.getIfBody()){	
				graphString += getSQLLopGraphString(sb);
			}
			for (StatementBlock sb : ifstmt.getElseBody()){	
				graphString += getSQLLopGraphString(sb);
			}
		}
		
		else {
			
			ArrayList<Hops> hopsDAG = current.get_hops();
			if (hopsDAG !=  null && hopsDAG.size() > 0) {
				Iterator<Hops> iter = hopsDAG.iterator();
				while (iter.hasNext()) {
					SQLLops h = iter.next().get_sqllops();

					String nodeList = new String("");
					nodeList = prepareSQLLopNodeList(h);
					graphString += nodeList;
				}
			}
		}
		
		return graphString;
	}
	
	private static String prepareSQLLopNodeList(SQLLops lop)
	{
		String s = new String("");

		if (lop.get_visited() == Hops.VISIT_STATUS.DONE)
			return s;
		
		String color = "white";
		if(lop.get_properties().getJoinType() == JOINTYPE.INNERJOIN)
			color = "wheat";
		else if(lop.get_properties().getJoinType() == JOINTYPE.LEFTJOIN)
			color = "lightblue";
		else if(lop.get_properties().getJoinType() == JOINTYPE.FULLOUTERJOIN)
			color = "gray";
		
		s += "    node" + lop.getId() + " [label=\"" + lop.get_tableName() + "\\nOp: " + lop.get_properties().getOpString() + " \\nJoin: " + lop.get_properties().getJoinType().toString() + "\\nAggOp: " + lop.get_properties().getAggType().toString()
		+ "\", color=" + color + " ]; \n";
		
		for (SQLLops l : lop.getInputs()) {
			String si = prepareSQLLopNodeList(l);
			s += si;

			String edge = "    node" + lop.getId() + " -> node" + l.getId() + "; \n";
			s += edge;
		}
		
		lop.set_visited(VISIT_STATUS.DONE);
		return s;
	}
}
