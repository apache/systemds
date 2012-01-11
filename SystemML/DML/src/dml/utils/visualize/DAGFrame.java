package dml.utils.visualize;

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
	
/*	    
	    draw = new JButton("Draw");
	    gbl.setConstraints(draw,gbc);
	    panel.add(draw);
	    draw.addActionListener(this);
	
	    layout = new JButton("Layout");
	    gbl.setConstraints(layout,gbc);
	    panel.add(layout);
	    layout.addActionListener(this);
	
	    printer = new JButton("Print");
	    gbl.setConstraints(printer,gbc);
	    panel.add(printer);
	    printer.addActionListener(this);
	
	    quit = new JButton("Quit");
	    gbl.setConstraints(quit,gbc);
	    panel.add(quit);
	    quit.addActionListener(this);
*/	
	    getContentPane().add("Center", jsp);
	    getContentPane().add("West", panel);
	
	    setVisible(true);
	    jsp.setViewportView(gp);
	}
	
	
	public void actionPerformed(ActionEvent evt) {
	    
		/*
		if(evt.getSource() instanceof JButton) {
			JButton tgt = (JButton)evt.getSource();
			if(tgt == draw) {
			    graph.repaint();
			} else if(tgt == quit) {
			    System.exit(0);
			} else if(tgt == printer) {
			    graph.printGraph(System.out);
			    System.out.flush();
			} else if(tgt == layout) {
			    Object connector = null;
			    try {
				connector = Runtime.getRuntime().exec(Demo12.SCRIPT);
			    } catch(Exception ex) {
				System.err.println("Exception while setting up Process: " + ex.getMessage() + "\nTrying URLConnection...");
				connector = null;
			    }
			    if(connector == null) {
				try {
				    connector = (new URL("http://www.research.att.com/~john/cgi-bin/format-graph")).openConnection();
				    URLConnection urlConn = (URLConnection)connector;
				    urlConn.setDoInput(true);
				    urlConn.setDoOutput(true);
				    urlConn.setUseCaches(false);
				    urlConn.setRequestProperty("Content-Type","application/x-www-form-urlencoded");
				} catch(Exception ex) {
				    System.err.println("Exception while setting up URLConnection: " + ex.getMessage() + "\nLayout not performed.");
				    connector = null;
				}
			    }
			    if(connector != null) {
				if(!GrappaSupport.filterGraph(graph,connector)) {
				    System.err.println("ERROR: somewhere in filterGraph");
				}
				if(connector instanceof Process) {
				    try {
					int code = ((Process)connector).waitFor();
					if(code != 0) {
					    System.err.println("WARNING: proc exit code is: " + code);
					}
				    } catch(InterruptedException ex) {
					System.err.println("Exception while closing down proc: " + ex.getMessage());
					ex.printStackTrace(System.err);
				    }
				}
				connector = null;
			    }
			    graph.repaint();
			}
	    }
	    */
	}
	
}
