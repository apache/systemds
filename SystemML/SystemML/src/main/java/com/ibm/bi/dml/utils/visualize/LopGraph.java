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
import com.ibm.bi.dml.lops.Aggregate;
import com.ibm.bi.dml.lops.Binary;
import com.ibm.bi.dml.lops.Data;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.Transform;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.ForStatement;
import com.ibm.bi.dml.parser.ForStatementBlock;
import com.ibm.bi.dml.parser.FunctionStatement;
import com.ibm.bi.dml.parser.FunctionStatementBlock;
import com.ibm.bi.dml.parser.IfStatement;
import com.ibm.bi.dml.parser.IfStatementBlock;
import com.ibm.bi.dml.parser.LanguageException;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.WhileStatement;
import com.ibm.bi.dml.parser.WhileStatementBlock;


public class LopGraph 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static String getString_dataLop(Lop lop) {
		String s = new String("");

		if ( lop.getOutputParameters().getLabel() != null ) {
			s += "    node" + lop.getID() + " [label=\"" + lop.getID() + ") " + lop.getType() + " " + lop.getOutputParameters().getLabel()
					+ " " + ((Data) lop).getOperationType();
		}
		else {
			s += "    node" + lop.getID() + " [label=\"" + lop.getID() + ") " + lop.getType() + " " + lop.getOutputParameters().getFile_name()
			+ " " + ((Data) lop).getOperationType();
		}
		if (((Data) lop).getOperationType() == com.ibm.bi.dml.lops.Data.OperationTypes.READ) {
			s += "\", color=wheat2 ]; \n";
		} else if (((Data) lop).getOperationType() == com.ibm.bi.dml.lops.Data.OperationTypes.WRITE) {
			s += "\", color=wheat4 ]; \n";
		}
		return s;
	}

	private static String getString_aggregateLop(Lop lop) {
		String s = new String("");
		s += "    node" + lop.getID() + " [label=\""  + lop.getID() + ") aggr";

		switch (((Aggregate) lop).getOperationType()) {
		case Sum:
			s += "(+)";
			break;
		case Product:
			s += "(*)";
			break;
		case Max:
			s += "(max)";
			break;
		case Min:
			s += "(min)";
			break;
		case Trace:
			s += "(trace)";
			break;
		default:
			s += ((Aggregate) lop).getOperationType();
			break;
		}
		s += "\", color=white ]; \n";
		return s;
	}

	private static String getString_transformLop(Lop lop) {
		String s = new String("");
		s += "    node" + lop.getID() + " [label=\""  + lop.getID() + ") transform";

		switch (((Transform) lop).getOperationType()) {
		case Transpose:
			s += "(t)";
			break;
		case Diag:
			s += "(diag)";
			break;
		case Reshape:
			s += "(reshape)";	
			break;
		default:
			s += ((Transform) lop).getOperationType();
			break;
		}
		s += "\", color=white ]; \n";
		return s;
	}

	private static String getString_binaryLop(Lop lop) {
		String s = new String("");
		s += "    node" + lop.getID() + " [label=\"" + lop.getID() + ") b";

		// Add,Subtract,Multiply,Divide,Max,Min
		switch (((Binary) lop).getOperationType()) {
		case ADD:
			s += "(+)";
			break;
		case SUBTRACT:
			s += "(-)";
			break;
		case MULTIPLY:
			s += "(*)";
			break;
		case DIVIDE:
			s += "(/)";
			break;
		case MAX:
			s += "(max)";
			break;
		case MIN:
			s += "(min)";
			break;
		default:
			s += ((Binary) lop).getOperationType();
			break;
		}
		s += "\", color=white ]; \n";
		return s;
	}

	private static String prepareLopsNodeList(Lop lop) {
		String s = new String("");

		if (lop.getVisited() == Lop.VisitStatus.DONE)
			return s;

		switch (lop.getType()) {
		case Data:
			s += getString_dataLop(lop);
			break;
		case Aggregate:
			s += getString_aggregateLop(lop);
			break;
		case Binary:
			s += getString_binaryLop(lop);
			break;
		case Grouping:
			s += "    node" + lop.getID() + " [label=\"" + lop.getID() + ") group\", color=white ]; \n";
			break;
		case Transform:
			s += getString_transformLop(lop);
			break;
		default:
			s += "    node" + lop.getID() + " [label=\"" + lop.getID() + ") " + lop.getType() + "\", color=white ]; \n";
			break;
		}

		for (int i = 0; i < lop.getInputs().size(); i++) {
			String si = prepareLopsNodeList(lop.getInputs().get(i));
			s += si;

			String edge = "    node" + lop.getID() + " -> node" + lop.getInputs().get(i).getID() + "; \n";
			s += edge;
		}
		lop.setVisited(Lop.VisitStatus.DONE);
		return s;
	}

	public static String getLopGraphString(DMLProgram dmlp, String title, int x, int y, String basePath) throws LanguageException {
		String graphString = new String("digraph G { \n node [ label = \"\\N\", style=filled ];  edge [dir=back]; \n ");
		
		for (String namespaceKey : dmlp.getNamespaces().keySet()){
			for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet()) {
				FunctionStatementBlock current = dmlp.getFunctionStatementBlock(namespaceKey, fname);
				graphString += getLopGraphString(current);
			}
		}
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			graphString += getLopGraphString(current);
		}
		graphString += "\n } \n";
		return graphString;
	}
			
	public static String getLopGraphString(StatementBlock current) {
	
		String graphString = new String();

		if (current instanceof WhileStatementBlock) {
			// Handle Predicate
			Hop predicateHops = ((WhileStatementBlock) current).getPredicateHops();
			if (predicateHops != null){
				String predicateString = prepareLopsNodeList(predicateHops.getLops());
				graphString += predicateString;
			}
			// handle children
			WhileStatement wstmt = (WhileStatement)((WhileStatementBlock)current).getStatement(0);
			for (StatementBlock sb : wstmt.getBody()){	
				graphString += getLopGraphString(sb);
			}
		}
		else if (current instanceof ForStatementBlock) {
			ForStatementBlock fsb = (ForStatementBlock) current;
			
			// Handle Predicate
			if (fsb.getFromLops() != null)
				graphString += prepareLopsNodeList(fsb.getFromLops());
			if (fsb.getToLops() != null)
				graphString += prepareLopsNodeList(fsb.getToLops());
			if (fsb.getIncrementLops() != null)
				graphString += prepareLopsNodeList(fsb.getIncrementLops());
			
			// handle children
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			for (StatementBlock sb : fstmt.getBody()){	
				graphString += getLopGraphString(sb);
			}
		}
		
		else if (current instanceof FunctionStatementBlock) {
			
			// handle children
			FunctionStatement fstmt = (FunctionStatement)((FunctionStatementBlock)current).getStatement(0);
			for (StatementBlock sb : fstmt.getBody()){	
				graphString += getLopGraphString(sb);
			}
		}
		
		else if (current instanceof IfStatementBlock) {
			// Handle Predicate
			Hop predicateHops = ((IfStatementBlock) current).getPredicateHops();
			String predicateString = prepareLopsNodeList(predicateHops.getLops());
			graphString += predicateString;
			
			// handle children
			IfStatement ifstmt = (IfStatement)((IfStatementBlock)current).getStatement(0);
			for (StatementBlock sb : ifstmt.getIfBody()){	
				graphString += getLopGraphString(sb);
			}
			for (StatementBlock sb : ifstmt.getElseBody()){	
				graphString += getLopGraphString(sb);
			}
		}
		
		else {
			ArrayList<Lop> lopsDAG = current.getLops();
			if (lopsDAG != null && lopsDAG.size() > 0) {
				Iterator<Lop> iter = lopsDAG.iterator();
				while (iter.hasNext()) {
					String nodeList = new String("");
					nodeList = prepareLopsNodeList(iter.next());
					graphString += nodeList;
				}
			}
		}

		return graphString;
	}
	
}
