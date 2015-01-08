/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.utils.visualize;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Window;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;

import com.ibm.bi.dml.utils.visualize.grappa.att.Graph;
import com.ibm.bi.dml.utils.visualize.grappa.att.GrappaAdapter;
import com.ibm.bi.dml.utils.visualize.grappa.att.GrappaPanel;

class DAGFrame extends JFrame implements ActionListener
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	GrappaPanel gp;
	Graph graph = null;
	
	JButton layout = null;
	JButton printer = null;
	JButton draw = null;
	JButton quit = null;
	JPanel panel = null;
	
	@SuppressWarnings("deprecation")
	public DAGFrame(Graph graph, String title, int x, int y) {
	    super(title);
	    this.graph = graph;
	
	    setSize(600,400);
	    setLocation(x,y);
	
	    addWindowListener(new WindowAdapter() {
		    public void windowClosing(WindowEvent wev) {
			Window w = wev.getWindow();
			w.setVisible(false);
			w.dispose();
			return;
		    }
		});
	
	    JScrollPane jsp = new JScrollPane();
	    jsp.getViewport().setBackingStoreEnabled(true);
	
	    gp = new GrappaPanel(graph);
	    gp.addGrappaListener(new GrappaAdapter());
	    gp.setScaleToFit(false);
	
	    // java.awt.Rectangle bbox = graph.getBoundingBox().getBounds();
	
	    GridBagLayout gbl = new GridBagLayout();
	    GridBagConstraints gbc = new GridBagConstraints();
	
	    gbc.gridwidth = GridBagConstraints.REMAINDER;
	    gbc.fill = GridBagConstraints.HORIZONTAL;
	    gbc.anchor = GridBagConstraints.NORTHWEST;
	
	    panel = new JPanel();
	    panel.setLayout(gbl);
	
	    getContentPane().add("Center", jsp);
	    getContentPane().add("West", panel);
	
	    setVisible(true);
	    jsp.setViewportView(gp);
	}
	
	
	public void actionPerformed(ActionEvent evt) {

	}
	
}
