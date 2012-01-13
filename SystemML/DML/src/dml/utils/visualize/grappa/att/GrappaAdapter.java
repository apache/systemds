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

import java.awt.*;
import java.awt.print.*;
import java.awt.geom.AffineTransform;
import java.awt.geom.Dimension2D;
import java.awt.geom.Rectangle2D;
import java.awt.event.InputEvent;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.util.*;

/**
 * A convenience class that implements the GrappaListener interface
 * for handling mouse-related activity that occurs on a graph.
 *
 * This particular GrappaListener implementation allows the following
 * interactions with a displayed graph:
 *
 * <UL>
 * <LI>
 * display tooltips for each graph element;
 * <LI>
 * button-1 click will select an element;
 * <LI>
 * button-1 sweep will select several elements;
 * <LI>
 * shift-button 1 click will create a node;
 * <LI>
 * shift-button 1 drag, when starting in one node and ending in another, will create an edge;
 * <LI>
 * button-2 or button-3 click will raise a pop-up option menu with:
 *   <UL>
 *   <LI>
 *   zoom in, zoom out and reset zoom options;
 *   <LI>
 *   a zoom-to-sweep option, if applicable;
 *   <LI>
 *   clear selection, enclose the selected items in a new subgraph, 
 *   preview deletion, cancel preview and perform deletion, if at least
 *   one item is selected;
 *   <LI>
 *   a tooltip on/off toggle option.
 *   </UL>
 * </UL>
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public class GrappaAdapter
    implements GrappaConstants, GrappaListener, ActionListener
{
  /**
   * The method called when a mouse click occurs on a displayed subgraph.
   *
   * @param subg displayed subgraph where action occurred
   * @param elem subgraph element in which action occurred
   * @param pt the point where the action occurred (graph coordinates)
   * @param modifiers mouse modifiers in effect
   * @param clickCount count of mouse clicks that triggered this action
   * @param panel specific panel where the action occurred
   */
    public void grappaClicked(Subgraph subg, Element elem, GrappaPoint pt, int modifiers, int clickCount, GrappaPanel panel) {
	if((modifiers&InputEvent.BUTTON1_MASK) == InputEvent.BUTTON1_MASK) {
	    if(clickCount == 1) {
		// looks like Java has a single click occur on the way to a
		// multiple click, so this code always executes (which is
		// not necessarily a bad thing)
		if(subg.getGraph().isSelectable()) {
		    if(modifiers == InputEvent.BUTTON1_MASK) {
			// select element
			if(elem == null) {
			    if(subg.currentSelection != null) {
				if(subg.currentSelection instanceof Element) {
				    ((Element)(subg.currentSelection)).highlight &= ~HIGHLIGHT_MASK;
				} else {
				    Vector vec = ((Vector)(subg.currentSelection));
				    for(int i = 0; i < vec.size(); i++) {
					((Element)(vec.elementAt(i))).highlight &= ~HIGHLIGHT_MASK;
				    }
				}
				subg.currentSelection = null;
				subg.getGraph().repaint();
			    }
			} else {
			    if(subg.currentSelection != null) {
				if(subg.currentSelection == elem) return;
				if(subg.currentSelection instanceof Element) {
				    ((Element)(subg.currentSelection)).highlight &= ~HIGHLIGHT_MASK;
				} else {
				    Vector vec = ((Vector)(subg.currentSelection));
				    for(int i = 0; i < vec.size(); i++) {
					((Element)(vec.elementAt(i))).highlight &= ~HIGHLIGHT_MASK;
				    }
				}
				subg.currentSelection = null;
			    }
			    elem.highlight |= SELECTION_MASK;
			    subg.currentSelection = elem;
			    subg.getGraph().repaint();
			}
		    } else if(modifiers == (InputEvent.BUTTON1_MASK|InputEvent.CTRL_MASK)) {
			// adjust selection
			if(elem != null) {
			    if((elem.highlight&SELECTION_MASK) == SELECTION_MASK) {
				// unselect element
				elem.highlight &= ~SELECTION_MASK;
				if(subg.currentSelection == null) {
				// something got messed up somewhere
				    throw new InternalError("currentSelection improperly maintained");
				} else if(subg.currentSelection instanceof Element) {
				    if(((Element)(subg.currentSelection)) != elem) {
					// something got messed up somewhere
					throw new InternalError("currentSelection improperly maintained");
				    }
				    subg.currentSelection = null;
				} else {
				    Vector vec = ((Vector)(subg.currentSelection));
				    boolean problem = true;
				    for(int i = 0; i < vec.size(); i++) {
					if(((Element)(vec.elementAt(i))) == elem) {
					    vec.removeElementAt(i);
					    problem = false;
					    break;
					}
				    }
				    if(problem) {
					// something got messed up somewhere
					throw new InternalError("currentSelection improperly maintained");
				    }
				}
			    } else {
				// select element
				elem.highlight |= SELECTION_MASK;
				if(subg.currentSelection == null) {
				    subg.currentSelection = elem;
				} else if(subg.currentSelection instanceof Element) {
				    Object obj = subg.currentSelection;
				    subg.currentSelection = new Vector();
				    ((Vector)(subg.currentSelection)).add(obj);
				    ((Vector)(subg.currentSelection)).add(elem);
				} else {
				    ((Vector)(subg.currentSelection)).add(elem);
				}
			    }
			    subg.getGraph().repaint();
			}
		    }
		}
	    } else {
		// multiple clicks
		// this code executes for each click beyond the first
		//System.err.println("clickCount="+clickCount);
	    }
	}
    }

  /**
   * The method called when a mouse press occurs on a displayed subgraph.
   *
   * @param subg displayed subgraph where action occurred
   * @param elem subgraph element in which action occurred
   * @param pt the point where the action occurred (graph coordinates)
   * @param modifiers mouse modifiers in effect
   * @param panel specific panel where the action occurred
   */
    public void grappaPressed(Subgraph subg, Element elem, GrappaPoint pt, int modifiers, GrappaPanel panel) {
	if((modifiers&(InputEvent.BUTTON2_MASK|InputEvent.BUTTON3_MASK)) != 0 && (modifiers&(InputEvent.BUTTON2_MASK|InputEvent.BUTTON3_MASK)) == modifiers) {
	    // pop-up menu if button2 or button3
	    javax.swing.JPopupMenu popup = new javax.swing.JPopupMenu();
	    javax.swing.JMenuItem item = null;
	    if(panel.getToolTipText() == null) {
		popup.add(item = new javax.swing.JMenuItem("ToolTips On"));
	    } else {
		popup.add(item = new javax.swing.JMenuItem("ToolTips Off"));
	    }
	    item.addActionListener(this);
	    popup.addSeparator();
	    popup.add(item = new javax.swing.JMenuItem("Print"));
	    item.addActionListener(this);
	    popup.addSeparator();
	    if(subg.currentSelection != null) {
		popup.add(item = new javax.swing.JMenuItem("Clear Selection"));
		item.addActionListener(this);
		popup.addSeparator();
		if(subg.currentSelection instanceof Element) {
		    popup.add(item = new javax.swing.JMenuItem("Select Siblings in Subgraph"));
		    item.addActionListener(this);
		    popup.addSeparator();
		}
		popup.add(item = new javax.swing.JMenuItem("Enclose Selected Items in a new Subgraph"));
		item.addActionListener(this);
		popup.addSeparator();
		popup.add(item = new javax.swing.JMenuItem("Preview Deletion"));
		item.addActionListener(this);
		popup.add(item = new javax.swing.JMenuItem("Cancel Preview"));
		item.addActionListener(this);
		popup.add(item = new javax.swing.JMenuItem("Perform Deletion"));
		item.addActionListener(this);
		popup.addSeparator();
	    }
	    if(panel.hasOutline()) {
		popup.add(item = new javax.swing.JMenuItem("Zoom to Sweep"));
		item.addActionListener(this);
	    }
	    popup.add(item = new javax.swing.JMenuItem("Zoom In"));
	    item.addActionListener(this);
	    popup.add(item = new javax.swing.JMenuItem("Zoom Out"));
	    item.addActionListener(this);
	    popup.add(item = new javax.swing.JMenuItem("Reset Zoom"));
	    item.addActionListener(this);
	    popup.add(item = new javax.swing.JMenuItem("Scale to Fit"));
	    item.addActionListener(this);
	    if(subg.hasEmptySubgraphs()) {
		popup.addSeparator();
		popup.add(item = new javax.swing.JMenuItem("Remove Empty Subgraphs"));
		item.addActionListener(this);
	    }

	    java.awt.geom.Point2D mpt = panel.getTransform().transform(pt,null);
	    popup.show(panel,(int)mpt.getX(),(int)mpt.getY());
	}
    }

  /**
   * The method called when a mouse release occurs on a displayed subgraph.
   *
   * @param subg displayed subgraph where action occurred
   * @param elem subgraph element in which action occurred
   * @param pt the point where the action occurred (graph coordinates)
   * @param modifiers mouse modifiers in effect
   * @param pressedElem subgraph element in which the most recent mouse press occurred
   * @param pressedPt the point where the most recent mouse press occurred (graph coordinates)
   * @param pressedModifiers mouse modifiers in effect when the most recent mouse press occurred
   * @param outline enclosing box specification from the previous drag position (for XOR reset purposes)
   * @param panel specific panel where the action occurred
   */
    public void grappaReleased(Subgraph subg, Element elem, GrappaPoint pt, int modifiers, Element pressedElem, GrappaPoint pressedPt, int pressedModifiers, GrappaBox outline, GrappaPanel panel) {
	if(modifiers == InputEvent.BUTTON1_MASK && subg.getGraph().isSelectable()) {
	    if(outline != null) {
		boolean xorOutline = false;
		if(subg.currentSelection != null) {
		    if(subg.currentSelection instanceof Element) {
			((Element)(subg.currentSelection)).highlight = 0;
		    } else {
			Vector vec = ((Vector)(subg.currentSelection));
			for(int i = 0; i < vec.size(); i++) {
			    ((Element)(vec.elementAt(i))).highlight = 0;
			}
		    }
		    subg.currentSelection = null;
		}
		Vector elems = GrappaSupport.findContainedElements(subg, outline);
		if(elems != null) {
		    drillDown(subg, elems, SELECTION_MASK, HIGHLIGHT_ON);
		    xorOutline = false;
		}
		if(!xorOutline) {
		    subg.getGraph().paintImmediately();
		}
		if(xorOutline) {
		    Graphics2D g2d = (Graphics2D)(panel.getGraphics());
		    AffineTransform orig = g2d.getTransform();
		    g2d.setTransform(panel.getTransform());
		    g2d.setXORMode(Color.darkGray);
		    g2d.draw(outline);
		    g2d.setPaintMode();
		    g2d.setTransform(orig);
		}
	    }
	} else if(modifiers == (InputEvent.BUTTON1_MASK|InputEvent.CTRL_MASK) && subg.getGraph().isSelectable()) {
	    if(outline != null) {
		Vector elems = GrappaSupport.findContainedElements(subg, outline);
		if(elems != null) {
		    drillDown(subg, elems, SELECTION_MASK, HIGHLIGHT_TOGGLE);
		    subg.getGraph().repaint();
		} else {
		    Graphics2D g2d = (Graphics2D)(panel.getGraphics());
		    AffineTransform orig = g2d.getTransform();
		    g2d.setTransform(panel.getTransform());
		    g2d.setXORMode(Color.darkGray);
		    g2d.draw(outline);
		    g2d.setPaintMode();
		    g2d.setTransform(orig);
		}
	    }
	} else if(modifiers == (InputEvent.BUTTON1_MASK|InputEvent.SHIFT_MASK) && subg.getGraph().isEditable()) {
	    if(elem != null && pressedElem != null) {
		if(pressedModifiers == modifiers) {
		    if(outline == null) {
			if(pressedElem == elem && pt.distance(pressedPt) < 5) {
			    // [should we only allow elem.isSubgraph()?]
			    // create node
			    Attribute[] attrs = null;
			    Attribute attr = subg.getNodeAttribute(LABEL_ATTR);
			    if(attr == null || attr.getValue().equals("\\N")) {
				attrs = new Attribute[] { new Attribute(Grappa.NODE, POS_ATTR, pt), new Attribute(Grappa.NODE, LABEL_ATTR, "Node" + subg.getGraph().getId(Grappa.NODE)) };
			    } else {
				attrs = new Attribute[] { new Attribute(Grappa.NODE, POS_ATTR, pt) };
			    }
			    Element el = subg.createElement(Grappa.NODE,null,attrs);
			    if(el != null) {
				el.buildShape();
				subg.getGraph().repaint();
			    }
			    subg.getGraph().repaint();
			} else if(pressedElem != elem && pressedElem.isNode() && elem.isNode()) {
			    // create edge
			    Object[] info = new Object[] { elem, null, pressedElem };
			    Attribute[] attrs = new Attribute[] { new Attribute(Grappa.EDGE, POS_ATTR, new GrappaLine(new GrappaPoint[] { ((Node)pressedElem).getCenterPoint(), ((Node)elem).getCenterPoint() }, subg.getGraph().isDirected()?GrappaLine.TAIL_ARROW_EDGE:GrappaLine.NONE_ARROW_EDGE) ) };
			    Element el = subg.createElement(Grappa.EDGE,info,attrs);
			    if(el != null) {
				el.buildShape();
				subg.getGraph().repaint();
			    }
			}
		    }
		}
	    }
	}
    }

  /**
   * The method called when a mouse drag occurs on a displayed subgraph.
   *
   * @param subg displayed subgraph where action occurred
   * @param currentPt the current drag point
   * @param currentModifiers the current drag mouse modifiers
   * @param pressedElem subgraph element in which the most recent mouse press occurred
   * @param pressedPt the point where the most recent mouse press occurred (graph coordinates)
   * @param pressedModifiers mouse modifiers in effect when the most recent mouse press occurred
   * @param outline enclosing box specification from the previous drag position (for XOR reset purposes)
   * @param panel specific panel where the action occurred
   */
    public void grappaDragged(Subgraph subg, GrappaPoint currentPt, int currentModifiers, Element pressedElem, GrappaPoint pressedPt, int pressedModifiers, GrappaBox outline, GrappaPanel panel) {
	if((currentModifiers&InputEvent.BUTTON1_MASK) == InputEvent.BUTTON1_MASK) {
	    if(currentModifiers == InputEvent.BUTTON1_MASK || currentModifiers == (InputEvent.BUTTON1_MASK|InputEvent.CTRL_MASK)) {
		Graphics2D g2d = (Graphics2D)(panel.getGraphics());
		AffineTransform orig = g2d.getTransform();
		g2d.setTransform(panel.getTransform());
		g2d.setXORMode(Color.darkGray);
		if(outline != null) {
		    g2d.draw(outline);
		}
		GrappaBox box = GrappaSupport.boxFromCorners(pressedPt.x, pressedPt.y, currentPt.x, currentPt.y);
		g2d.draw(box);
		g2d.setPaintMode();
		g2d.setTransform(orig);
	    }
	}
    }

  /**
   * The method called when a element tooltip is needed.
   *
   * @param subg displayed subgraph where action occurred
   * @param elem subgraph element in which action occurred
   * @param pt the point where the action occurred (graph coordinates)
   * @param modifiers mouse modifiers in effect
   * @param panel specific panel where the action occurred
   *
   * @return the tip to be displayed or null; in this implementation,
   * if the mouse is in a graph element that
   * has its <I>tip</I> attribute defined, then that text is returned.
   * If that attribute is not set, the element name is returned.
   * If the mouse is outside the graph bounds, then the text supplied
   * to the graph setToolTipText method is supplied.
   */
    public String grappaTip(Subgraph subg, Element elem, GrappaPoint pt, int modifiers, GrappaPanel panel) {
	String tip = null;

	if(elem == null) {
	    if((tip = panel.getToolTipText()) == null) {
		if((tip = subg.getGraph().getToolTipText()) == null) {
		    tip = Grappa.getToolTipText();
		}
	    }
	} else {
	    switch(elem.getType()) {
	    case SUBGRAPH:
		Subgraph sg = (Subgraph)elem;
		if((tip = (String)sg.getAttributeValue(TIP_ATTR)) == null) {
		    if(subg.getShowSubgraphLabels()) {
			tip = sg.getName();
		    } else {
			if((tip = (String)sg.getAttributeValue(LABEL_ATTR)) == null) {
			    tip = sg.getName();
			}
		    }
		    tip = "Subgraph: " + tip;
		}
		break;
	    case EDGE:
		Edge edge = (Edge)elem;
		if((tip = (String)edge.getAttributeValue(TIP_ATTR)) == null) {
		    if(subg.getShowEdgeLabels()) {
			tip = edge.toString();
		    } else {
			if((tip = (String)edge.getAttributeValue(LABEL_ATTR)) == null) {
			    tip = edge.toString();
			}
		    }
		    tip = "Edge: " + tip;
		}
		break;
	    case NODE:
		Node node = (Node)elem;
		if((tip = (String)node.getAttributeValue(TIP_ATTR)) == null) {
		    if(subg.getShowNodeLabels()) {
			tip = node.getName();
		    } else {
			if((tip = (String)node.getAttributeValue(LABEL_ATTR)) == null || tip.equals("\\N")) {
			    tip = node.getName();
			}
		    }
		    tip = "Node: " + tip;
		}
		break;
	    default:
		throw new RuntimeException("unexpected type (" + elem.getType() + ")");
	    }
	}

	return(tip);
    }

    ///////////////////////////////////////////////////////////////////////////
    //
    // ActionListener
    //
    ///////////////////////////////////////////////////////////////////////////

    /**
     * Invoked when an action occurs.
     *
     * @param aev the action event trigger.
     */
    public void actionPerformed(ActionEvent aev) {
	Object src = aev.getSource();
	if(src instanceof javax.swing.JMenuItem) {
	    Object parent = ((javax.swing.JMenuItem)src).getParent();
	    if(parent instanceof javax.swing.JPopupMenu) {
		Object invoker = ((javax.swing.JPopupMenu)(((javax.swing.JMenuItem)src).getParent())).getInvoker();
		if(invoker instanceof GrappaPanel) {
		    GrappaPanel gp = (GrappaPanel)invoker;
		    Subgraph subg = gp.getSubgraph();
		    String text = ((javax.swing.JMenuItem)src).getText();
		    if(text.startsWith("Cancel")) {
		        if(subg.currentSelection == null) return;
			if(subg.currentSelection instanceof Element) {
			    GrappaSupport.setHighlight((Element)(subg.currentSelection), DELETION_MASK, HIGHLIGHT_OFF); 
			} else {
			    Vector vec = (Vector)(subg.currentSelection);
			    for(int i = 0; i < vec.size(); i++) {
				GrappaSupport.setHighlight((Element)(vec.elementAt(i)), DELETION_MASK, HIGHLIGHT_OFF); 
			    }
			}
			subg.getGraph().repaint();
		    } else if(text.startsWith("Clear")) {
		        if(subg.currentSelection == null) return;
			if(subg.currentSelection instanceof Element) {
			    GrappaSupport.setHighlight((Element)(subg.currentSelection), 0, HIGHLIGHT_OFF); 
			} else {
			    Vector vec = (Vector)(subg.currentSelection);
			    for(int i = 0; i < vec.size(); i++) {
				GrappaSupport.setHighlight((Element)(vec.elementAt(i)), 0, HIGHLIGHT_OFF); 
			    }
			}
			subg.currentSelection = null;
			subg.getGraph().repaint();
		    } else if(text.startsWith("Select")) {
		        if(subg.currentSelection == null) return;
			if(!(subg.currentSelection instanceof Element)) return;
			Element elem = ((Element)subg.currentSelection).getSubgraph();
			if(elem == null || subg.currentSelection == elem || !(elem instanceof Subgraph)) return;
			((Element)(subg.currentSelection)).highlight &= ~HIGHLIGHT_MASK;
			Vector elems = new Vector();
			Enumeration enm = ((Subgraph)elem).nodeElements();
			while(enm.hasMoreElements())
			    elems.add(enm.nextElement());
			enm = ((Subgraph)elem).edgeElements();
			while(enm.hasMoreElements())
			    elems.add(enm.nextElement());
			subg.currentSelection = null;
			if(elems != null && elems.size() > 0)
			    drillDown(subg, elems, SELECTION_MASK, HIGHLIGHT_ON);
			subg.getGraph().repaint();
		    } else if(text.startsWith("Enclose")) {
		        if(subg.currentSelection == null || subg.currentSelection == subg) return;
			Subgraph newsubg = new Subgraph(subg);
			if(subg.currentSelection instanceof Element) {
			    GrappaSupport.setHighlight((Element)(subg.currentSelection), 0, HIGHLIGHT_OFF); 
			    ((Element)(subg.currentSelection)).setSubgraph(newsubg);
			} else {
			    Vector vec = (Vector)(subg.currentSelection);
			    for(int i = 0; i < vec.size(); i++) {
				GrappaSupport.setHighlight((Element)(vec.elementAt(i)), 0, HIGHLIGHT_OFF); 
				if(((Element)(vec.elementAt(i))) == subg) continue;
				((Element)(vec.elementAt(i))).setSubgraph(newsubg);
			    }
			}
			subg.currentSelection = null;
			subg.getGraph().repaint();
		    } else if(text.startsWith("Preview")) {
		        if(subg.currentSelection == null) return;
			if(subg.currentSelection instanceof Element) {
			    GrappaSupport.setHighlight((Element)(subg.currentSelection), DELETION_MASK, HIGHLIGHT_ON); 
			} else {
			    Vector vec = (Vector)(subg.currentSelection);
			    for(int i = 0; i < vec.size(); i++) {
				GrappaSupport.setHighlight((Element)(vec.elementAt(i)), DELETION_MASK, HIGHLIGHT_ON); 
			    }
			}
			subg.getGraph().repaint();
		    } else if(text.startsWith("Perform")) {
		        if(subg.currentSelection == null) return;
			if(subg.currentSelection instanceof Element) {
			    ((Element)(subg.currentSelection)).delete();
			} else {
			    Vector vec = (Vector)(subg.currentSelection);
			    for(int i = 0; i < vec.size(); i++) {
				((Element)(vec.elementAt(i))).delete();
			    }
			}
			subg.currentSelection = null;
			subg.getGraph().repaint();
		    } else if(text.startsWith("Remove")) {
			subg.removeEmptySubgraphs();
		    } else if(text.startsWith("Reset")) {
			gp.setScaleToFit(false);
			gp.setScaleToSize(null);
			gp.resetZoom();
			gp.clearOutline();
			gp.repaint();
		    } else if(text.startsWith("Scale")) {
			gp.setScaleToFit(true);
			gp.repaint();
		    } else if(text.startsWith("Print")) {
			PageFormat pf = new PageFormat();
			Rectangle2D bb = subg.getBoundingBox();
			if(bb.getWidth() > bb.getHeight())
			    pf.setOrientation(PageFormat.LANDSCAPE);
			try {
			    PrinterJob printJob = PrinterJob.getPrinterJob();
			    printJob.setPrintable(gp, pf);
			    if (printJob.printDialog()) {
				printJob.print();
			    }
			} catch (Exception ex) {
			    Grappa.displayException(ex, "Problem with print request");
			}
		    } else if(text.startsWith("ToolTips")) {
			if(text.indexOf("Off") > 0) {
			    gp.setToolTipText(null);
			} else {
			    String tip = subg.getGraph().getToolTipText();
			    if(tip == null) {
				tip = Grappa.getToolTipText();
			    }
			    gp.setToolTipText(tip);
			}
		    } else if(text.startsWith("Zoom In")) {
			gp.setScaleToFit(false);
			gp.setScaleToSize(null);
			gp.multiplyScaleFactor(1.25);
			gp.clearOutline();
			gp.repaint();
		    } else if(text.startsWith("Zoom Out")) {
			gp.setScaleToFit(false);
			gp.setScaleToSize(null);
			gp.multiplyScaleFactor(0.8);
			gp.clearOutline();
			gp.repaint();
		    } else if(text.startsWith("Zoom to")) {
		        //if(subg.currentSelection == null) return;
			gp.setScaleToFit(false);
			gp.setScaleToSize(null);
			gp.zoomToOutline();
			gp.clearOutline();
			gp.repaint();
		    }
		}
	    }
	}
    }

    ///////////////////////////////////////////////////////////////////////////

    protected void drillDown(Subgraph subg, Vector elems, int mode, int setting) {
	Object obj = null;
	for(int i = 0; i < elems.size(); i++) {
	    obj = elems.elementAt(i);
	    if(obj instanceof Vector) {
		drillDown(subg, (Vector)obj, mode, setting);
	    } else {
		GrappaSupport.setHighlight(((Element)obj), mode, setting);
		switch(setting) {
		case HIGHLIGHT_TOGGLE:
		    if((((Element)obj).highlight&mode) == mode) {
			if(subg.currentSelection == null) {
			    subg.currentSelection = obj;
			} else if(subg.currentSelection instanceof Element) {
			    Object crnt = subg.currentSelection;
			    subg.currentSelection = new Vector();
			    ((Vector)(subg.currentSelection)).add(crnt);
			    ((Vector)(subg.currentSelection)).add(obj);
			} else {
			    ((Vector)(subg.currentSelection)).add(obj);
			}
		    } else {
			if(subg.currentSelection == obj) {
			    subg.currentSelection = null;
			} else if(subg.currentSelection instanceof Vector) {
			    ((Vector)(subg.currentSelection)).remove(obj);
			}
		    }
		    break;
		case HIGHLIGHT_ON:
		    if(subg.currentSelection == null) {
			subg.currentSelection = obj;
		    } else if(subg.currentSelection instanceof Element) {
			Object crnt = subg.currentSelection;
			subg.currentSelection = new Vector();
			((Vector)(subg.currentSelection)).add(crnt);
			((Vector)(subg.currentSelection)).add(obj);
		    } else {
			((Vector)(subg.currentSelection)).add(obj);
		    }
		    break;
		case HIGHLIGHT_OFF:
		    if(subg.currentSelection != null) {
			if(subg.currentSelection == obj) {
			    subg.currentSelection = null;
			} else if(subg.currentSelection instanceof Vector) {
			    ((Vector)(subg.currentSelection)).remove(obj);
			}
		    }
		    break;
		}
	    }
	}
    }
}
