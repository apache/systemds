/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.utils.visualize;

import java.util.ArrayList;
import java.util.Iterator;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
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


public class HopGraph 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static String getString_dataHop(Hop h) {
		String s = new String("");
		// assert h.getKind() == Hops.Kind.DataOp
		if (((DataOp) h).get_dataop() == Hop.DataOpTypes.PERSISTENTREAD) {
			s += "    node" + h.getHopID() + " [label=\"" + h.getHopID() + ")" + h.get_name() + " " + h.getOpString()
					+ "\", color=royalblue2 ]; \n";
		} else if (((DataOp) h).get_dataop() == Hop.DataOpTypes.TRANSIENTREAD) {
			s += "    node" + h.getHopID() + " [label=\"" + h.getHopID() + ")" + h.get_name() + " " + h.getOpString()
					+ "\", color=wheat2 ]; \n";
		}
		// Persistent/Transient Writes
		else if (((DataOp) h).get_dataop() == Hop.DataOpTypes.PERSISTENTWRITE) {
			s += "    node" + h.getHopID() + " [label=\"" + h.getHopID() + ")" + h.get_name() + " " + h.getOpString()
					+ "\", color=orangered2 ]; \n";
		} else if (((DataOp) h).get_dataop() == Hop.DataOpTypes.TRANSIENTWRITE) {
			s += "    node" + h.getHopID() + " [label=\"" + h.getHopID() + ")" + h.get_name() + " " + h.getOpString()
					+ "\", color=wheat4 ]; \n";
		}
		return s;
	}

	private static String getString_sinkHop(Hop h) {
		// h is the final result.. so, no parents
		// asseert h.getParent().size() == 0
		String s = new String("");
		if (h.getKind() == Hop.Kind.DataOp) {
			String sdata = getString_dataHop(h);
			s += sdata;
		} else {
			s += "    node" + h.getHopID() + " [label=\"" + h.getHopID() + ")" + h.get_name() + " " + h.getOpString()
					+ "\", color=lightcoral ]; \n";
		}

		return s;
	}

	private static String getString_sourceHop(Hop h) {
		// asseert h.getInput().size() == 0
		String s = new String("");
		if (h.getKind() == Hop.Kind.DataOp) {
			String sdata = getString_dataHop(h);
			s += sdata;
		} else {
			s += "    node" + h.getHopID() + " [label=\"" + h.getHopID() + ")" + h.get_name() + " " + h.getOpString()
					+ "\", color=lavender ]; \n";
		}
		return s;
	}

	private static String prepareHopsNodeList(Hop h) {
		String s = new String("");

		if (h.get_visited() == Hop.VISIT_STATUS.DONE)
			return s;

		if (h.getInput().size() == 0) {
			s += getString_sourceHop(h);
		} else if (h.getParent().size() == 0) {
			s += getString_sinkHop(h);
		} else {
			if (h.getKind() == Hop.Kind.DataOp) {
				s += getString_dataHop(h);
			} else {
				s += "    node" + h.getHopID() + " [label=\"" + h.getHopID() + ")" + h.get_name() + " " + h.getOpString()
						+ "\", color=white ]; \n";
			}
		}

		for (Hop hi : h.getInput()) {
			String si = prepareHopsNodeList(hi);
			s += si;

			String edge = "    node" + h.getHopID() + " -> node" + hi.getHopID() + "; \n";
			s += edge;
		}

		/*
		 * for ( Hops hp : h.getParent() ) { if ( hp.get_visited() ==
		 * Hops.VISIT_STATUS.NOTVISITED) { String edge = "    node" +
		 * h.getHopID() + " -> node" + hp.getHopID() + "; \n"; s += edge; } }
		 */
		h.set_visited(Hop.VISIT_STATUS.DONE);

		return s;
	}
	
	public static String getHopGraphString(DMLProgram dmlp, String title, int x, int y, String basePath) throws HopsException, LanguageException {
		
		
		String graphString = new String("digraph G { \n node [ label = \"\\N\", style=filled ]; edge [dir=back]; \n ");
		
		for (String namespaceKey : dmlp.getNamespaces().keySet()){
			for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet()){
				FunctionStatement fstmt = (FunctionStatement)dmlp.getFunctionStatementBlock(namespaceKey,fname).getStatement(0);
				for (StatementBlock sb : fstmt.getBody()){
					graphString += getHopGraphString(sb);
				}
			}
		}
		
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			graphString += getHopGraphString(current);
		}
		
		graphString += "\n } \n";
		return graphString;
	}
	
	public static String getHopGraphString(StatementBlock current) throws HopsException {
	
		String graphString = new String();
		
		if (current instanceof WhileStatementBlock) {
			// Handle Predicate
			
			Hop predicateHops = ((WhileStatementBlock) current).getPredicateHops();
			if (predicateHops != null){
				String predicateString = prepareHopsNodeList(predicateHops);
				graphString += predicateString;
			}
			// handle children
			WhileStatement wstmt = (WhileStatement)((WhileStatementBlock)current).getStatement(0);
			for (StatementBlock sb : wstmt.getBody()){	
				graphString += getHopGraphString(sb);
			}
			
		}
		else if (current instanceof ForStatementBlock) {
			// Handle Predicate
			ForStatementBlock fsb = (ForStatementBlock) current;
			if (fsb.getFromHops() != null)
				graphString += prepareHopsNodeList(fsb.getFromHops());
			if (fsb.getToHops() != null)
				graphString += prepareHopsNodeList(fsb.getToHops());
			if (fsb.getIncrementHops() != null)
				graphString += prepareHopsNodeList(fsb.getIncrementHops());
			
			// handle children
			ForStatement fstmt = (ForStatement)((ForStatementBlock)current).getStatement(0);
			for (StatementBlock sb : fstmt.getBody()){	
				graphString += getHopGraphString(sb);
			}
			
		}
		
		else if (current instanceof IfStatementBlock) {
			// Handle Predicate
			Hop predicateHops = ((IfStatementBlock) current).getPredicateHops();
			String predicateString = prepareHopsNodeList(predicateHops);
			graphString += predicateString;
			
			// handle children
			IfStatement ifstmt = (IfStatement)((IfStatementBlock)current).getStatement(0);
			for (StatementBlock sb : ifstmt.getIfBody()){	
				graphString += getHopGraphString(sb);
			}
			for (StatementBlock sb : ifstmt.getElseBody()){	
				graphString += getHopGraphString(sb);
			}
		}
		
		else {
			
			ArrayList<Hop> hopsDAG = current.get_hops();
			if (hopsDAG !=  null && hopsDAG.size() > 0) {
				Iterator<Hop> iter = hopsDAG.iterator();
				while (iter.hasNext()) {
					Hop h = iter.next();

					String nodeList = new String("");
					nodeList = prepareHopsNodeList(h);
					graphString += nodeList;
				}
			}
		}
				
		return graphString;
	}
	
}
