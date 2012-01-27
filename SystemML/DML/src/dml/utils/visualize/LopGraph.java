package dml.utils.visualize;

import java.util.ArrayList;
import java.util.Iterator;

import dml.hops.Hops;
import dml.lops.Aggregate;
import dml.lops.Binary;
import dml.lops.Data;
import dml.lops.Lops;
import dml.lops.Transform;
import dml.parser.DMLProgram;
import dml.parser.ForStatement;
import dml.parser.ForStatementBlock;
import dml.parser.FunctionStatement;
import dml.parser.FunctionStatementBlock;
import dml.parser.IfStatement;
import dml.parser.IfStatementBlock;
import dml.parser.StatementBlock;
import dml.parser.WhileStatement;
import dml.parser.WhileStatementBlock;
import dml.utils.LanguageException;

public class LopGraph {

	private static String getString_dataLop(Lops lop) {
		String s = new String("");

		if ( lop.getOutputParameters().getLabel() != null ) {
			s += "    node" + lop.getID() + " [label=\"" + lop.getID() + ") " + lop.getType() + " " + lop.getOutputParameters().getLabel()
					+ " " + ((Data) lop).getOperationType();
		}
		else {
			s += "    node" + lop.getID() + " [label=\"" + lop.getID() + ") " + lop.getType() + " " + lop.getOutputParameters().getFile_name()
			+ " " + ((Data) lop).getOperationType();
		}
		if (((Data) lop).getOperationType() == dml.lops.Data.OperationTypes.READ) {
			s += "\", color=wheat2 ]; \n";
		} else if (((Data) lop).getOperationType() == dml.lops.Data.OperationTypes.WRITE) {
			s += "\", color=wheat4 ]; \n";
		}
		return s;
	}

	private static String getString_aggregateLop(Lops lop) {
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

	private static String getString_transformLop(Lops lop) {
		String s = new String("");
		s += "    node" + lop.getID() + " [label=\""  + lop.getID() + ") transform";

		switch (((Transform) lop).getOperationType()) {
		case Transpose:
			s += "(t)";
			break;
		case VectortoDiagMatrix:
			s += "(DiagV2M)";
			break;
		default:
			s += ((Transform) lop).getOperationType();
			break;
		}
		s += "\", color=white ]; \n";
		return s;
	}

	private static String getString_binaryLop(Lops lop) {
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

	private static String prepareLopsNodeList(Lops lop) {
		String s = new String("");

		if (lop.get_visited() == Lops.VISIT_STATUS.DONE)
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
			String si = prepareLopsNodeList((Lops) lop.getInputs().get(i));
			s += si;

			String edge = "    node" + lop.getID() + " -> node" + lop.getInputs().get(i).getID() + "; \n";
			s += edge;
		}
		lop.set_visited(Lops.VISIT_STATUS.DONE);
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
			Hops predicateHops = ((WhileStatementBlock) current).getPredicateHops();
			if (predicateHops != null){
				String predicateString = prepareLopsNodeList(predicateHops.get_lops());
				graphString += predicateString;
			}
			// handle children
			WhileStatement wstmt = (WhileStatement)((WhileStatementBlock)current).getStatement(0);
			for (StatementBlock sb : wstmt.getBody()){	
				graphString += getLopGraphString(sb);
			}
		}
		else if (current instanceof ForStatementBlock) {
			// Handle Predicate
			Hops predicateHops = ((ForStatementBlock) current).getPredicateHops();
			String predicateString = prepareLopsNodeList(predicateHops.get_lops());
			graphString += predicateString;
			
			// handle children
			ForStatement fstmt = (ForStatement)((ForStatementBlock)current).getStatement(0);
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
		
		
		else if (current instanceof ForStatementBlock) {
			// Handle Predicate
			Hops predicateHops = ((ForStatementBlock) current).getPredicateHops();
			String predicateString = prepareLopsNodeList(predicateHops.get_lops());
			graphString += predicateString;
			
			// handle children
			ForStatement fstmt = (ForStatement)((ForStatementBlock)current).getStatement(0);
			for (StatementBlock sb : fstmt.getBody()){	
				graphString += getLopGraphString(sb);
			}
		}
		else if (current instanceof IfStatementBlock) {
			// Handle Predicate
			Hops predicateHops = ((IfStatementBlock) current).getPredicateHops();
			String predicateString = prepareLopsNodeList(predicateHops.get_lops());
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
			ArrayList<Lops> lopsDAG = current.get_lops();
			if (lopsDAG != null && lopsDAG.size() > 0) {
				Iterator<Lops> iter = lopsDAG.iterator();
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
