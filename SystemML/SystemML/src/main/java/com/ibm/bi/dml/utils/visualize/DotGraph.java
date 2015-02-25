/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.utils.visualize;

import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.LanguageException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.Program;

import com.ibm.bi.dml.utils.visualize.grappa.att.Graph;
import com.ibm.bi.dml.utils.visualize.grappa.att.GrappaSupport;

public class DotGraph 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	public DotGraph() {

	}

	private void printGraphString( String str, String title ) {
		System.out.println("---- " + title + " in DOT format ----");
		System.out.println(str);
		System.out.println("--------------------------------");
	}
	
	@SuppressWarnings("deprecation")
	void visualize(String content, String title, int x, int y, String basePath, boolean draw_graph) {
		String osname = System.getProperty("os.name"); 
		if (osname.contains("Windows")) {
			//printGraphString(content, title);
			if ( draw_graph ) {
				Graph graph = GrappaSupport.filterGraph(content, basePath);
				DAGFrame dagFrame = new DAGFrame(graph, title, x, y);
				dagFrame.show();
			}
		}
		else {
			if ( draw_graph ) { 
				System.out.println("Warning: Can not visualize the graph on current OS (" + osname + "). Displaying the graph in text format.");
				printGraphString(content, title);
			}
		}
	}

	/**
	 * 
	 * @param dmlp
	 *            -- DML Program whose HopsDAG is visualized
	 * @param title
	 *            -- title of the HopsDAG that is displayed on the JFrame
	 * @param x
	 *            -- x-coordinate of the window in which DAG is displayed
	 * @param y
	 *            -- y-coordinate of the window in which DAG is displayed
	 *            Coordinates are introduced so that all windows (JFrames) don't
	 *            get displayed on top of each other
	 * @param basePath
	 *            -- the path of DML source directory. If dml source is at
	 *            /path/to/dml/src then basePath should be /path/to/dml.
	 * @throws HopsException 
	 * @throws LanguageException 
	 */

	public void drawHopsDAG(DMLProgram dmlp, String title, int x, int y, String basePath, boolean draw_graph) throws HopsException, LanguageException {
		String graphString = HopGraph.getHopGraphString(dmlp, title, x, y, basePath);
		visualize(graphString, title, x, y, basePath, draw_graph);
	}
	
	public void drawSQLLopsDAG(DMLProgram dmlp, String title, int x, int y, String basePath, boolean draw_graph) throws HopsException, LanguageException {
		String graphString = SQLLopGraph.getSQLLopGraphString(dmlp, title, x, y, basePath);
		visualize(graphString, title, x, y, basePath, draw_graph);
	}

	/**
	 * 
	 * @param dmlp
	 *            -- DML Program whose LopsDAG is visualized
	 * @param title
	 *            -- title of the LopsDAG that is displayed on the JFrame
	 * @param x
	 *            -- x-coordinate of the window in which DAG is displayed
	 * @param y
	 *            -- y-coordinate of the window in which DAG is displayed
	 *            Coordinates are introduced so that all windows (JFrames) don't
	 *            get displayed on top of each other
	 * @param basePath
	 *            -- the path of DML source directory. If dml source is at
	 *            /path/to/dml/src then basePath should be /path/to/dml.
	 * @throws LanguageException 
	 */
	public void drawLopsDAG(DMLProgram dmlp, String title, int x, int y, String basePath, boolean draw_graph) throws LanguageException {
		String graphString = LopGraph.getLopGraphString(dmlp, title, x, y, basePath);
		visualize(graphString, title, x, y, basePath, draw_graph);
	}
	
	public void drawInstructionsDAG(Program dmlp, String title, int x, int y, String basePath, boolean draw_graph) throws DMLUnsupportedOperationException {
		String graphString = InstructionGraph.getInstructionGraphString(dmlp, title, x, y, basePath);
		visualize(graphString, title, x, y, basePath, draw_graph);
	}

}
