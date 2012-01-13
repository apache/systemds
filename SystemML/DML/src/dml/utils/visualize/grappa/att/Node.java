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

import java.util.*;
import java.io.*;

import java.awt.Color;
import java.awt.geom.Point2D;


/**
 * This class describes a node.
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public class Node extends Element
{
    /**
     * Default node name prefix used by setName().
     *
     * @see setName()
     */
    public final static String defaultNamePrefix = "N";

    // vector of edges going into the node
    private Vector inEdges = null;
    // vector of edges going out of the node
    private Vector outEdges = null;

    // vector of edge ports (not used yet)
    private Vector Ports = null;

    /**
     * Use this constructor when creating a node within a subgraph.
     *
     * @param subg the parent subgraph.
     * @param name the name of this node.
     */
    public Node(Subgraph subg, String name) {
	super(Grappa.NODE,subg);
	setName(name);

	nodeAttrsOfInterest();
    }

    /**
     * Use this constructor when creating a node within a subgraph
     * with an automatically generated name.
     *
     * @param subg the parent subgraph.
     * @see setName()
     */
    public Node(Subgraph subg) {
	this(subg,(String)null);
    }

    // a listing of the attributes of interest for Nodes
    private void nodeAttrsOfInterest() {
	attrOfInterest(DISTORTION_ATTR);
	attrOfInterest(HEIGHT_ATTR);
	attrOfInterest(ORIENTATION_ATTR);
	attrOfInterest(PERIPHERIES_ATTR);
	attrOfInterest(POS_ATTR);
	attrOfInterest(SIDES_ATTR);
	attrOfInterest(SKEW_ATTR);
	attrOfInterest(STYLE_ATTR);
	attrOfInterest(WIDTH_ATTR);
    }

    // override Element methods

    /**
     * Check if this element is a node.
     * Useful for testing the subclass type of an Element object.
     *
     * @return true if this object is a Node.
     */
    public boolean isNode() {
	return(true);
    }

    /**
     * Get the type of this element.
     * Useful for distinguishing among Element objects.
     *
     * @return the class variable constant Grappa.NODE
     * @see Grappa#NODE
     */
    public int getType() {
	return(Grappa.NODE);
    }

    /**
     * Generates and sets the name for this node.
     * The generated name is the concatenation of Node.defaultNamePrefix
     * with the numeric id of this node instance.
     *
     * @see Node#defaultNamePrefix
     * @see Element#getId()
     */
    void setName() {
	String oldName = name;
    
	while(true) {
	    name = Node.defaultNamePrefix + getId() + "_" + System.currentTimeMillis();
	    if(getGraph().findNodeByName(name) == null) {
		break;
	    }
	}

	// update subgraph node dictionary
	if(oldName != null) {
	    getSubgraph().removeNode(oldName);
	}
	getSubgraph().addNode(this);

	canonName = null;
	resetEdgeNames();
    }

    /**
     * Sets the node name to the supplied argument.
     * When the argument is null, setName() is called.
     *
     * @exception IllegalArgumentException when newName is not unique.
     * @param newName the new name for the node.
     * @see Node#setName()
     */
    public void setName(String newName) throws IllegalArgumentException {
	if(newName == null) {
	    setName();
	    return;
	}

	String oldName = name;
    
	// test if name is the same as the old name (if any)
	if(oldName != null && oldName.equals(newName)) {
	    return;
	}

	// is name unique?
	if(getGraph().findNodeByName(newName) != null) {
	    throw new IllegalArgumentException("node name (" + newName + ") is not unique");
	}

	// update subgraph node dictionary
	if(oldName != null) {
	    getSubgraph().removeNode(oldName);
	}
	name = newName;
	getSubgraph().addNode(this);

	canonName = null;
	resetEdgeNames();
    }

    private void resetEdgeNames() {
	Edge edge;

	if (inEdges != null) {
	    for(int i = 0; i < inEdges.size(); i++) {
		edge = (Edge)(inEdges.elementAt(i));
		edge.canonName = null;
	    }
	}
	if (outEdges != null) {
	    for(int i = 0; i < outEdges.size(); i++) {
		edge = (Edge)(outEdges.elementAt(i));
		edge.canonName = null;
	    }
	}
    }

    /**
     * Add the given edge to this node's inEdges or outEdges dictionaries,
     * if it is not already there.
     * The boolean indicates whether the edge terminates at (inEdge) or
     * emanates from (outEdge) the node.
     *
     * @param edge the edge to be added to this node's dictionary.
     * @param inEdge if set true, add to inEdges dictionary otherwise add
     *               to outEdges dictionary.
     * @see Edge
     */
    synchronized public void addEdge(Edge edge, boolean inEdge) {
	if(edge == null) return;
	if(inEdge) {
	    if(inEdges == null) {
		inEdges = new Vector();
	    }
	    if(!inEdges.contains(edge)) {
		inEdges.addElement(edge);
	    }
	} else {
	    if(outEdges == null) {
		outEdges = new Vector();
	    }
	    outEdges.addElement(edge);
	    if(!outEdges.contains(edge)) {
		outEdges.addElement(edge);
	    }
	}
    }

    /**
     * Find an outbound edge given its head and key.
     *
     * @param head the Node at the head of the edge
     * @param key the key String associated with the edge
     *
     * @return the matching edge or null
     */
    public Edge findOutEdgeByKey(Node head, String key) {
	if(head == null || key == null || outEdges == null) {
	    return null;
	}
	Edge edge = null;
	for(int i = 0; i < outEdges.size(); i++) {
	    edge = (Edge)(outEdges.elementAt(i));
	    if(head == edge.getHead() && key.equals(edge.getKey())) {
		return edge;
	    }
	}
	return null;
    }

    /**
     * Find an inbound edge given its tail and key.
     *
     * @param tail the Node at the tail of the edge
     * @param key the key String associated with the edge
     *
     * @return the matching edge or null
     */
    public Edge findInEdgeByKey(Node tail, String key) {
	if(tail == null || key == null || inEdges == null) {
	    return null;
	}
	Edge edge = null;
	for(int i = 0; i < inEdges.size(); i++) {
	    edge = (Edge)(inEdges.elementAt(i));
	    if(tail == edge.getTail() && key.equals(edge.getKey())) {
		return edge;
	    }
	}
	return null;
    }

    /**
     * Returns the center point of the node.
     * as determined from the height
     * and width attributes of the node.
     *
     * @return the node's center point.
     */
    public GrappaPoint getCenterPoint() {
	GrappaPoint pt = (GrappaPoint)getAttributeValue(POS_ATTR);
	if(pt == null) { // this should never be null, but just in case...
	    pt = new GrappaPoint();
	} else if(!Grappa.centerPointNodes) {
	    Double w = (Double)getAttributeValue(WIDTH_ATTR);
	    Double h = (Double)getAttributeValue(HEIGHT_ATTR);
	    if(w != null && h != null) { // these should never be null, but...
		pt = new GrappaPoint(pt.x - (w.doubleValue()/2.0), pt.y - (h.doubleValue()/2.0));
	    }
	} else {
	    pt = new GrappaPoint(pt.x, pt.y); // return copy
	}
	return(pt);
    }

    /**
     * Remove the given edge from this node's inEdges or outEdges dictionaries.
     * The boolean indicates whether the edge terminates at (inEdge) or
     * emanates from (outEdge) the node.
     *
     * @param edge the edge to be removed from this node's dictionary.
     * @param inEdge if set true, remove from inEdges dictionary otherwise
     *               remove from outEdges dictionary.
     * @see Edge
     */
    synchronized public void removeEdge(Edge edge, boolean inEdge) {
	if(edge == null) return;
	if(inEdge) {
	    if(inEdges == null) return;
	    inEdges.removeElement(edge);
	} else {
	    if(outEdges == null) return;
	    outEdges.removeElement(edge);
	}
    }

    /**
     * Print the node description to the provided stream.
     *
     * @param out the output text stream for writing the description.
     */
    public void printNode(PrintWriter out) {
	this.printElement(out);
    }

    /**
     * Returns the attribute conversion type for the supplied attribute name.
     * After node specific attribute name/type mappings are checked, mappings
     * at the element level are checked.
     *
     * @param attrname the attribute name
     * @return the currently associated attribute type
     */
    public static int attributeType(String attrname) {
	int convtype = -1;
	int hashCode;

	if(attrname != null) {
	    hashCode = attrname.hashCode();

	    if(hashCode == DISTORTION_HASH && attrname.equals(DISTORTION_ATTR)) {
		convtype = DOUBLE_TYPE;
	    } else if(hashCode == ORIENTATION_HASH && attrname.equals(ORIENTATION_ATTR)) {
		convtype = DOUBLE_TYPE;
	    } else if(hashCode == PERIPHERIES_HASH && attrname.equals(PERIPHERIES_ATTR)) {
		convtype = INTEGER_TYPE;
	    } else if(hashCode == POS_HASH && attrname.equals(POS_ATTR)) {
		convtype = POINT_TYPE;
	    } else if(hashCode == SHAPE_HASH && attrname.equals(SHAPE_ATTR)) {
		convtype = SHAPE_TYPE;
	    } else if(hashCode == SIDES_HASH && attrname.equals(SIDES_ATTR)) {
		convtype = INTEGER_TYPE;
	    } else if(hashCode == SKEW_HASH && attrname.equals(SKEW_ATTR)) {
		convtype = DOUBLE_TYPE;
	    } else {
		return(Element.attributeType(attrname));
	    }
	}

	return(convtype);
    }

    /**
     * Get an Enumeration of the edges directed to or from this node.
     *
     * @return an Enumeration of all the edges (in or out) associated with this node.
     */
    public Enumeration edgeElements() {
	return new Enumerator(inEdges, outEdges);
    }

    /**
     * Get an Enumeration of the edges directed to this node.
     *
     * @return an Enumeration of all the inbound edges associated with this node.
     */
    public Enumeration inEdgeElements() {
	return new Enumerator(inEdges, null);
    }

    /**
     * Get an Enumeration of the edges directed from this node.
     *
     * @return an Enumeration of all the outbound edges associated with this node.
     */
    public Enumeration outEdgeElements() {
	return new Enumerator(null, outEdges);
    }

    class Enumerator implements Enumeration {
	int inCnt = 0;
	int outCnt = 0;
	Vector inEdges = null;
	Vector outEdges = null;

	Enumerator(Vector inEdges, Vector outEdges) {
	    inCnt = (inEdges == null) ? 0 : inEdges.size();
	    outCnt = (outEdges == null) ? 0 : outEdges.size();
	    this.inEdges = inEdges;
	    this.outEdges = outEdges;
	}
  
	public boolean hasMoreElements() {
	    int tmp;
	    if(inCnt > 0 && inCnt > (tmp = inEdges.size())) inCnt = tmp;
	    if(outCnt > 0 && outCnt > (tmp = outEdges.size())) outCnt = tmp;
	    return((inCnt+outCnt) > 0);
	}

	public Object nextElement() {
	    synchronized (Node.this) {
		int tmp;
		if(inCnt > 0 && inCnt > (tmp = inEdges.size())) inCnt = tmp;
		if(inCnt > 0) {
		    return inEdges.elementAt(--inCnt);
		}
		if(outCnt > 0 && outCnt > (tmp = outEdges.size())) outCnt = tmp;
		if(outCnt > 0) {
		    return outEdges.elementAt(--outCnt);
		}
		throw new NoSuchElementException("Node$Enumerator");
	    }
	}
    }
}
