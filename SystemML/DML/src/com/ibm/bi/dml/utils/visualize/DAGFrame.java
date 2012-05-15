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

import dml.utils.visualize.grappa.att.Graph;
import dml.utils.visualize.grappa.att.GrappaAdapter;
import dml.utils.visualize.grappa.att.GrappaPanel;

class DAGFrame extends JFrame implements ActionListener
{
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
			System.exit(0);
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
