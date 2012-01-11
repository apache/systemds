/*
 *  This software may only be used by you under license from AT&T Corp.
 *  ("AT&T").  A copy of AT&T's Source Code Agreement is available at
 *  AT&T's Internet website having the URL:
 *  <http://www.research.att.com/sw/tools/graphviz/license/source.html>
 *  If you received this software without first entering into a license
 *  with AT&T, you have an infringing copy of this software and cannot use
 *  it without violating AT&T's intellectual property rights.
 */
package dml.utils.visualize.grappa.att;

import java.io.*;
import java.awt.*;
import java.awt.event.*;

/**
 * A class for displaying exception information in a pop-up frame.
 * As a convenience, an instance exists as a static member of
 * the <code>Grappa</code> class.
 *
 * @see Grappa#displayException(java.lang.Exception)
 * @see Grappa#displayException(java.lang.Exception,java.lang.String)
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public class ExceptionDisplay {
    private String title = null;
    Exception exception = null;
    Display display = null;


    /**
     * Creates an instance of the class for displaying exceptions.
     *
     * @param title the title for the pop-up frame
     */
    public ExceptionDisplay(String title) {
	this.title = title;
    }

    /**
     * Pops up the frame and displays information on the supplied exception.
     * Initially, a text area displays the message associated with the exception.
     * By pressing a button, an end-user can view a stack trace as well.
     *
     * @param ex the exception about which informtaion is to be displayed.
     */
    public void displayException(Exception ex) {
	displayException(ex,null);
    }

    /**
     * Pops up the frame and displays information on the supplied exception.
     * Initially, a text area displays the supplied string followed on the
     * next line by the message associated with the exception.
     * By pressing a button, an end-user can view a stack trace as well.
     *
     * @param ex the exception about which informtaion is to be displayed.
     */
    public void displayException(Exception ex, String msg) {
	if(display == null) display = new Display(title);
	exception = ex;
	if(ex == null && msg == null) {
	    return;
	}
	if(msg != null) {
	    if(ex == null) {
		display.setText(msg);
	    } else {
		display.setText(msg + Grappa.NEW_LINE + ex.getMessage());
	    }
	} else {
	    display.setText(ex.getMessage());
	}
	display.setVisible(true);
    }

    // TODO: re-do this using JFrame (not a big deal)
    class Display extends Frame {
	private TextArea textarea = null;
	private Panel buttonPanel = null;
	private Button trace = null;
	private Button dismiss = null;
	private WindowObserver observer = null;

	Display(String title) {
	    super(title);

	    observer = new WindowObserver();

	    GridBagLayout gbl = new GridBagLayout();
	    GridBagConstraints gbc = new GridBagConstraints();
	    gbc.insets = new Insets(4,4,4,4);
	    gbc.weightx = 1;
	    gbc.weighty = 1;
	    gbc.gridwidth = GridBagConstraints.REMAINDER;
	    setLayout(gbl);

	    textarea = new TextArea("",7,80);
	    textarea.setEditable(false);

	    buttonPanel = new Panel();
	    buttonPanel.setLayout(new BorderLayout());

	    trace = new Button("Stack Trace");
	    trace.addActionListener(observer);
	    dismiss = new Button("Dismiss");
	    dismiss.addActionListener(observer);

	    buttonPanel.add("West",trace);
	    buttonPanel.add("East",dismiss);

	    gbc.fill = GridBagConstraints.BOTH;
	    gbl.setConstraints(textarea,gbc);
	    add(textarea);
	    gbc.weighty = 0;
	    gbc.fill = GridBagConstraints.HORIZONTAL;
	    gbl.setConstraints(buttonPanel,gbc);
	    add(buttonPanel);

	    addWindowListener(observer);
	    pack();
	}

	void setText(String text) {
	    if(text == null) text = "No message to display, try stack trace.";
	    textarea.setText(text);
	}

	Exception getException() {
	    return exception;
	}

	class WindowObserver extends WindowAdapter implements ActionListener {

	    public void windowClosing(WindowEvent evt) {
		dismiss();
	    }

	    private void dismiss() {
		setVisible(false);
		dispose();
		display = null;
	    }

	    public void actionPerformed(ActionEvent evt) {
		Object src = evt.getSource();
		if(src instanceof Button) {
		    Button btn = (Button)src;
		    if(btn.getLabel().equals("Dismiss")) {
			setVisible(false);
		    } else if(btn.getLabel().equals("Stack Trace")) {
			if(getException() == null) {
			    setText("No stack trace available (exception is null).");
			} else {
			    StringWriter swriter = new StringWriter();
			    PrintWriter pwriter = new PrintWriter(swriter);
			    getException().printStackTrace(pwriter);
			    pwriter.flush();
			    setText(swriter.toString());
			    pwriter.close();
			}
		    }
		}
	    }
	}
    }
}
