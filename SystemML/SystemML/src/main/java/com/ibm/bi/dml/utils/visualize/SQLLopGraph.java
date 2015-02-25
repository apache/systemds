/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.utils.visualize;

import java.util.ArrayList;
import java.util.Iterator;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.VisitStatus;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.ForStatement;
import com.ibm.bi.dml.parser.ForStatementBlock;
import com.ibm.bi.dml.parser.FunctionStatement;
import com.ibm.bi.dml.parser.IfStatement;
import com.ibm.bi.dml.parser.IfStatementBlock;
import com.ibm.bi.dml.parser.LanguageException;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.WhileStatement;
import com.ibm.bi.dml.parser.WhileStatementBlock;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.JOINTYPE;


public class SQLLopGraph 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
			SQLLops predicateSQLLop = ((WhileStatementBlock) current).getPredicateHops().getSqlLops();
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
			ForStatementBlock fsb = (ForStatementBlock) current;
			if (fsb.getFromHops() != null)
				graphString += prepareSQLLopNodeList(fsb.getFromHops().getSqlLops());
			if (fsb.getToHops() != null)
				graphString += prepareSQLLopNodeList(fsb.getToHops().getSqlLops());
			if (fsb.getIncrementHops() != null)
				graphString += prepareSQLLopNodeList(fsb.getIncrementHops().getSqlLops());
			
			// handle children
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			for (StatementBlock sb : fstmt.getBody()){	
				graphString += getSQLLopGraphString(sb);
			}
			
		}
		
		else if (current instanceof IfStatementBlock) {
			// Handle Predicate
			SQLLops predicateSQLLop = ((IfStatementBlock) current).getPredicateHops().getSqlLops();
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
			
			ArrayList<Hop> hopsDAG = current.get_hops();
			if (hopsDAG !=  null && hopsDAG.size() > 0) {
				Iterator<Hop> iter = hopsDAG.iterator();
				while (iter.hasNext()) {
					SQLLops h = iter.next().getSqlLops();

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

		if (lop.getVisited() == Hop.VisitStatus.DONE)
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
		
		lop.setVisited(VisitStatus.DONE);
		return s;
	}
}
