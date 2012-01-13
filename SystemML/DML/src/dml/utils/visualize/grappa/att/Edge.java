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

/**
 * This class describes an edge.
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public class Edge extends Element
{
    /**
     * Default edge name prefix used by setName().
     *
     * @see Edge#setName()
     */
    public final static String defaultNamePrefix = "E";

    /*
     * end nodes (port Ids are not used yet)
     */
    private Node headNode;
    private String headPortId = null;
    private Node tailNode;
    private String tailPortId = null;
    private String key = null;

    /*
     * direction info (adjusted here and by GrappaNexus)
     */
    int direction = GrappaLine.NONE_ARROW_EDGE;

    /**
     * Use this constructor when creating an edge.
     *
     * @param subg the parent subgraph.
     * @param tail node anchoring the tail of the edge.
     * @param head node anchoring the head of the edge.
     */
    public Edge(Subgraph subg, Node tail, Node head) {
	this(subg,tail,null,head,null,null,null);
    }

    /**
     * Use this constructor when creating an edge with ports.
     *
     * @param subg the parent subgraph.
     * @param tail node anchoring the tail of the edge.
     * @param tailPort the port to use within the tail node.
     * @param head node anchoring the head of the edge.
     * @param headPort the port to use within the head node.
     */
    public Edge(Subgraph subg, Node tail, String tailPort, Node head, String headPort) {
	this(subg,tail,tailPort,head,headPort,null,null);
    }

    /**
     * Use this constructor when creating an edge requiring a key to distinguish it.
     *
     * @param subg the parent subgraph.
     * @param tail node anchoring the tail of the edge.
     * @param tailPort the port to use within the tail node.
     * @param head node anchoring the head of the edge.
     * @param headPort the port to use within the head node.
     * @param key identifier (used in conjection with tail/head, but not ports) to uniquely define edge (and prevent unwanted duplicate from being created)
     */
    public Edge(Subgraph subg, Node tail, String tailPort, Node head, String headPort, String key) throws RuntimeException {
	this(subg,tail,tailPort,head,headPort,key,null);
    }

    /**
     * Use this constructor when creating an edge with a supplied unique name for easy look-up (the name is also used as the key).
     *
     * @param subg the parent subgraph.
     * @param tail node anchoring the tail of the edge.
     * @param head node anchoring the head of the edge.
     * @param name identifier to uniquely define edge within the entire graph (reather than just between head/tail pairs)
     */
    public Edge(Subgraph subg, Node tail, Node head, String name) throws RuntimeException {
	this(subg,tail,null,head,null,null,name);
    }

    /**
     * Use this constructor when creating an edge requiring a key to distinguish it and a supplied lookup name.
     * When name is null, it is automatically generated. When key is null, it is automatically generated or set to name, if it was supplied.
     *
     * @param subg the parent subgraph.
     * @param tail node anchoring the tail of the edge.
     * @param tailPort the port to use within the tail node.
     * @param head node anchoring the head of the edge.
     * @param headPort the port to use within the head node.
     * @param key identifier (used in conjection with tail/head, but not ports) to uniquely define edge (and prevent unwanted duplicate from being created)
     * @param name a unique name that can be used for lookup (if null, automatically generated)
     */
    public Edge(Subgraph subg, Node tail, String tailPort, Node head, String headPort, String key, String name) throws RuntimeException {
	super(Grappa.EDGE,subg);
	boolean directed = subg.getGraph().isDirected();

	if(directed)
	    direction = GrappaLine.TAIL_ARROW_EDGE;
	else
	    direction = GrappaLine.NONE_ARROW_EDGE;

	if(subg.getGraph().isStrict()) {
	    if(tail == head) {
		throw new RuntimeException("cannot create self-looping edge in a strict graph (" + tail.getName() + (directed?"->":"--") + head.getName() + ")");
	    } else {
		Enumeration enm = Edge.findEdgesByEnds(tail,head);
		if(enm.hasMoreElements()) {
		    if(!directed) {
			throw new RuntimeException("cannot create multiple edges between the same nodes in a strict graph");
		    } else {
			Edge tmpedge = null;
			while(enm.hasMoreElements()) {
			    tmpedge = (Edge)enm.nextElement();
			    if(tmpedge.getHead() == head && tmpedge.getTail() == tail) {
				throw new RuntimeException("cannot create multiple edges between the same nodes in the same direction in a strict directed graph");
			    }
			}
		    }
		}
	    }
	}
	if(!directed && tail.getId() > head.getId()) {
	    Node tmpNode = tail;
	    tail = head;
	    head = tmpNode;
	    String tmpPort = tailPort;
	    tailPort = headPort;
	    headPort = tmpPort;
	}
	tailNode = tail;
	if(tailPort != null) {
	    tailPortId = new String(tailPort);
	}
	headNode = head;
	if(headPort != null) {
	    headPortId = new String(headPort);
	}
	if(name != null) {
	    if(subg.getGraph().findEdgeByName(name) != null) {
		throw new RuntimeException("cannot create edge with duplicate name '" + name + "' (" + tailNode.getName() + " -> " + headNode.getName() + ")");
	    }
	    this.name = name;
	    subg.addEdge(this);
	    if(key == null) {
		key = name;
	    }
	} else {
	    setName();
	}
	if(key == null) {
	    if(headPort != null && tailPort != null) {
		this.key = tailPort + "::" + headPort;
	    } else if(headPort != null) {
		this.key = "::" + headPort;
	    } else if(tailPort != null) {
		this.key = tailPort + "::";
	    } else {
		this.key = name;
	    }
	} else {
	    this.key = key;
	}
	if(this.key != null) {
	    if(findEdgeByKey(tailNode,headNode,this.key) != null) {
		subg.removeEdge(this.name);
		throw new RuntimeException("cannot create duplicate edge (" + tailNode.getName() + (directed?"->":"--") + headNode.getName() + ") with key '" + this.key + "'");
	    }
	}
	tailNode.addEdge(this,false);
	headNode.addEdge(this,true);

	edgeAttrsOfInterest();
    }

    // a listing of the attributes of interest for Edges
    private void edgeAttrsOfInterest() {
	attrOfInterest(POS_ATTR);
	attrOfInterest(DIR_ATTR);
	attrOfInterest(LP_ATTR);
	attrOfInterest(HEADLABEL_ATTR);
	attrOfInterest(HEADLP_ATTR);
	attrOfInterest(TAILLABEL_ATTR);
	attrOfInterest(TAILLP_ATTR);
	attrOfInterest(STYLE_ATTR);
    }

    /**
     * Returns the edge with the given tail node, head node and key.
     *
     * @param tail the tail node of the desired edge.
     * @param head the head node of the desired edge.
     * @param key the key specifying the desired edge.
     * @return the Edge matching the arguments or null, if there is no match.
     * @see Edge#findEdgesByEnds
     */
    public static Edge findEdgeByKey(Node tail, Node head, String key) {
	if(tail == null || head == null || key == null) {
	    return(null);
	}
	return tail.findOutEdgeByKey(head,key);
    }

    /**
     * Check if this element is an edge.
     * Useful for testing the subclass type of a Element object.
     *
     * @return true if this object is a Edge.
     */
    public boolean isEdge() {
	return(true);
    }

    /**
     * Get the type of this element.
     * Useful for distinguishing Element objects.
     *
     * @return the class variable constant Grappa.EDGE.
     * @see Grappa
     */
    public int getType() {
	return(Grappa.EDGE);
    }

    /**
     * Generates and sets the name for this edge.
     * The generated name is the concatenation of tail node name,
     * the separator ">>", the head node name, the separator "##",
     * and the id of this edge Instance.
     * Also, takes the opportunity to add the edge to the subgraph and node
     * dictionaries.
     * Implements the abstract Element method.
     *
     * @see Element#getId()
     */
    void setName() {
	String oldName = name;
    
	while(true) {
	    name = Edge.defaultNamePrefix + getId() + "_" + System.currentTimeMillis();
	    if(getGraph().findEdgeByName(name) == null) {
		break;
	    }
	}

	// update subgraph edge dictionary
	if(oldName != null) {
	    getSubgraph().removeEdge(oldName);
	}
	getSubgraph().addEdge(this);

	canonName = null;
    }

    /**
     * Get the key for this edge.
     *
     * @return the key of the edge
     */
    public String getKey() {
	return key;
    }

    /**
     * Get the node at the head end of the edge.
     *
     * @return the head node of the edge
     */
    public Node getHead() {
	return headNode;
    }

    /**
     * Get the head port id of the edge.
     *
     * @return the head port id of the edge
     */
    public String getHeadPortId() {
	return headPortId;
    }

    /**
     * Get the node at the tail end of the edge.
     *
     * @return the tail node of the edge
     */
    public Node getTail() {
	return tailNode;
    }

    /**
     * Get the tail port id of the edge.
     *
     * @return the tail port id of the edge
     */
    public String getTailPortId() {
	return tailPortId;
    }

    /**
     * Get the String rendition of the edge.
     *
     * @return the string rendition of the edge, quoted as needed.
     */
    public String toString() {
	if(canonName == null) {
	    String tail = null;
	    String head = null;

	    if(tailPortId == null) {
		tail = tailNode.toString();
	    } else {
		tail = tailNode.toString() + ":" + canonString(tailPortId);
	    }
	    if(headPortId == null) {
		head = headNode.toString();
	    } else {
		head = headNode.toString() + ":" + canonString(headPortId);
	    }

	    if(getGraph().isDirected()) {
		canonName = tail + " -> " + head;
	    } else {
		canonName = tail + " -- " + head;
	    }
	}
	return(canonName);
    }

    /**
     * Print the edge description to the provided stream.
     *
     * @param out the output stream for writing the description.
     */
    public void printEdge(PrintWriter out) {
	this.printElement(out);
    }

    /**
     * Check if the edge connects in the forward direction.
     *
     * @return true when edge connects in the forward direction (tail to head)
     */
    public boolean goesForward() {
	return(direction != GrappaLine.HEAD_ARROW_EDGE);
    }

    /**
     * Check if the edge connects in the reverse direction.
     *
     * @return true when edge connects in the reverse direction (head to tail)
     */
    public boolean goesReverse() {
	return(direction != GrappaLine.TAIL_ARROW_EDGE);
    }


    /**
     * Returns the attribute conversion type for the supplied attribute name.
     * After edge specific attribute name/type mappings are checked, mappings
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

	    if(hashCode == POS_HASH && attrname.equals(POS_ATTR)) {
		convtype = LINE_TYPE;
	    } else if(hashCode == MINLEN_HASH && attrname.equals(MINLEN_ATTR)) {
		convtype = INTEGER_TYPE;
	    } else if(hashCode == DIR_HASH && attrname.equals(DIR_ATTR)) {
		convtype = DIR_TYPE;
	    } else if(hashCode == WEIGHT_HASH && attrname.equals(WEIGHT_ATTR)) {
		convtype = DOUBLE_TYPE;
	    } else if(hashCode == HEADLABEL_HASH && attrname.equals(HEADLABEL_ATTR)) {
		convtype = STRING_TYPE;
	    } else if(hashCode == HEADLP_HASH && attrname.equals(HEADLP_ATTR)) {
		convtype = POINT_TYPE;
	    } else if(hashCode == TAILLABEL_HASH && attrname.equals(TAILLABEL_ATTR)) {
		convtype = STRING_TYPE;
	    } else if(hashCode == TAILLP_HASH && attrname.equals(TAILLP_ATTR)) {
		convtype = POINT_TYPE;
	    } else {
		return(Element.attributeType(attrname));
	    }
	}

	return(convtype);
    }

    /**
     * Returns an enumeration of edges that have one end fixed at node1
     * and the other end at node2.  If node2 is empty, an enumeration of
     * all edges attached to node1 is returned.
     *
     * @param node1 one vertex of the set of edges to be returned
     * @param node2 the other vertex of the set of edges to be returned,
     *              or null for no constraint on the other vertex
     * @return an enumeration of Edge objects.
     */
    public static Enumeration findEdgesByEnds(Node node1, Node node2) {
	if(node1 == null) {
	    return Grappa.emptyEnumeration.elements();
	}
	return new Enumerator(node1,node2);
    }

    static class Enumerator implements Enumeration {
	Node node1 = null;
	Node node2 = null;
	Edge next = null;
	Enumeration outEdges = null;
	Enumeration inEdges = null;

	Enumerator(Node node1, Node node2) {
	    this.node1 = node1;
	    this.node2 = node2;
	    if(node1 != null) {
		this.outEdges = node1.outEdgeElements();
		this.inEdges = node1.inEdgeElements();
		next = getNext();
	    }
	}

	private Edge getNext() {
	    Edge tmpEdge = null;
	    if(outEdges != null) {
		while(outEdges.hasMoreElements()) {
		    tmpEdge = (Edge)outEdges.nextElement();
		    if(node2 == null || tmpEdge.getHead() == node2) {
			return tmpEdge;
		    }
		}
		outEdges = null;
	    }
	    if(inEdges != null) {
		while(inEdges.hasMoreElements()) {
		    tmpEdge = (Edge)inEdges.nextElement();
		    if(node2 == null || tmpEdge.getTail() == node2) {
			return tmpEdge;
		    }
		}
		inEdges = null;
	    }
	    return null;
	}
  
	public boolean hasMoreElements() {
	    return(next != null);
	}

	public Object nextElement() {
	    if(next == null) {
		throw new NoSuchElementException("Node$Enumerator");
	    }
	    Edge edge = next;
	    next = getNext();
	    return edge;
	}
    }
}
