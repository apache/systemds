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
 * This class is the root of the overall graph and provides methods for
 * working with the entire graph (for example, printing the graph). It is an
 * extension of the Subgraph class.
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public class Graph extends Subgraph
{
    /**
     * The string used for indentation when printing out the graph.
     */
    public final static String INDENT_STRING = "  ";

    /**
     * When filtering a graph (see GrappaSupport.filterGraph) or whenever
     * edge names are desired as part of the output of a printGraph call,
     * set this value to true. The edge name will be included among
     * edge attributes in the print output as '__nAmE__'. filterGraph
     * automatically handles setting this variable, but users writing
     * their own version of filterGraph will need it.
     */
    public boolean filterMode = false;

    // used with getIndent, incrementIndent and decrementIndent
    private StringBuffer indent = null;

    // used for error message (when set by setErrorWriter)
    private PrintWriter errWriter = null;

    // for keeping track of paint calls
    private boolean paintCalled = false;
    // for indicating if graph is busy being painted or altered
    private boolean busy = false;
    // for indicating if synchronization (involving the above) should be used
    private boolean synchronizePaint = false;

    //private UndoStack undoStack = new UndoStack();
    //private EditOp lastUndo = null;

    // graph-specific outside-the-bounds tooltip text
    private String toolTipText = null;

    // list of panels displaying this graph
    private List panelList = null;

    // counters for subgraph, node and edge elements (for id generation)
    private int gid = 0;
    private int nid = 0;
    private int eid = 0;


    // indicators for properties of the graph
    private boolean editable = false;
    private boolean menuable = false;
    private boolean selectable = true;

    // not used yet
    // public boolean autoUpdate = false; // TODO: add thread, etc.

    // directed graph?
    private boolean directed = true;

    // strict graph?
    private boolean strict = false;

    // for mapping id to an element
    Hashtable id2element = null;

    // Grappa global attributes (apply to all elements)
    private Hashtable grattributes = null;

    // tables for graph default node, edge and graph attributes, which are
    // initialized below
    private static Hashtable sysdfltNodeAttributes  = new Hashtable(8);
    private static Hashtable sysdfltEdgeAttributes  = new Hashtable(7);
    private static Hashtable sysdfltGraphAttributes = new Hashtable(11);


    // graph default node, edge and graph attributes, these should be
    // consistent with the dot layout program, although it is not necessary.
    static {
	// node
	putAttribute(sysdfltNodeAttributes,NODE,COLOR_ATTR,"black");
	putAttribute(sysdfltNodeAttributes,NODE,FONTCOLOR_ATTR,"black");
	putAttribute(sysdfltNodeAttributes,NODE,FONTNAME_ATTR,"TimesRoman");
	putAttribute(sysdfltNodeAttributes,NODE,FONTSIZE_ATTR,"14");
	putAttribute(sysdfltNodeAttributes,NODE,FONTSTYLE_ATTR,"normal");
	putAttribute(sysdfltNodeAttributes,NODE,HEIGHT_ATTR,"0.5");
	putAttribute(sysdfltNodeAttributes,NODE,POS_ATTR,"0,0");
	putAttribute(sysdfltNodeAttributes,NODE,LABEL_ATTR,"\\N");
	putAttribute(sysdfltNodeAttributes,NODE,SHAPE_ATTR,"ellipse");
	putAttribute(sysdfltNodeAttributes,NODE,STYLE_ATTR,GrappaStyle.DEFAULT_SET_STRING);
	putAttribute(sysdfltNodeAttributes,NODE,WIDTH_ATTR,"0.75");

	// edge
	putAttribute(sysdfltEdgeAttributes,EDGE,COLOR_ATTR,"black");
	putAttribute(sysdfltEdgeAttributes,EDGE,DIR_ATTR,"forward");
	putAttribute(sysdfltEdgeAttributes,EDGE,FONTCOLOR_ATTR,"black");
	putAttribute(sysdfltEdgeAttributes,EDGE,FONTNAME_ATTR,"TimesRoman");
	putAttribute(sysdfltEdgeAttributes,EDGE,FONTSIZE_ATTR,"14");
	putAttribute(sysdfltEdgeAttributes,EDGE,FONTSTYLE_ATTR,"normal");
	putAttribute(sysdfltEdgeAttributes,EDGE,MINLEN_ATTR,"1");
	putAttribute(sysdfltEdgeAttributes,EDGE,STYLE_ATTR,GrappaStyle.DEFAULT_SET_STRING);
	putAttribute(sysdfltEdgeAttributes,EDGE,WEIGHT_ATTR,"1");

	// graph
	putAttribute(sysdfltGraphAttributes,SUBGRAPH,CLUSTERRANK_ATTR,"local");
	putAttribute(sysdfltGraphAttributes,SUBGRAPH,COLOR_ATTR,"white");
	putAttribute(sysdfltGraphAttributes,SUBGRAPH,FONTCOLOR_ATTR,"black");
	putAttribute(sysdfltGraphAttributes,SUBGRAPH,FONTNAME_ATTR,"TimesRoman");
	putAttribute(sysdfltGraphAttributes,SUBGRAPH,FONTSIZE_ATTR,"14");
	putAttribute(sysdfltGraphAttributes,SUBGRAPH,FONTSTYLE_ATTR,"normal");
	putAttribute(sysdfltGraphAttributes,SUBGRAPH,MARGIN_ATTR,"0.5,0.5");
	putAttribute(sysdfltGraphAttributes,SUBGRAPH,MCLIMIT_ATTR,"1");
	putAttribute(sysdfltGraphAttributes,SUBGRAPH,NODESEP_ATTR,"0.25");
	putAttribute(sysdfltGraphAttributes,SUBGRAPH,ORIENTATION_ATTR,"portrait");
	putAttribute(sysdfltGraphAttributes,SUBGRAPH,RANKDIR_ATTR,"TB");
	putAttribute(sysdfltGraphAttributes,SUBGRAPH,RANKSEP_ATTR,"0.75");
	putAttribute(sysdfltGraphAttributes,SUBGRAPH,STYLE_ATTR,GrappaStyle.DEFAULT_SET_STRING);
    }

    // used for the above static initialization
    private static void putAttribute(Hashtable table, int type, String name, String value) {
	Attribute attr = new Attribute(type,name,value);
	attr.clearChanged();
	table.put(name,attr);
    }

    /**
     * Reference FontRenderContext
     */
    public final java.awt.font.FontRenderContext REFCNTXT = new java.awt.font.FontRenderContext(IDENTXFRM, Grappa.useAntiAliasing, Grappa.useFractionalMetrics);

    /**
     * Creates a new, empty Graph object.
     *
     * @param graphName the name of this graph.
     * @param directed use true if graph is to be a directed graph
     * @param strict use true if graph is a strict graph
     */
    public Graph(String graphName, boolean directed, boolean strict) {
	//super();
	initialize(graphName);

	setDirection(directed);
	this.strict = strict;

	// grappa attributes used for drawing
	setGrappaAttribute(GRAPPA_BACKGROUND_COLOR_ATTR,"white");
	setGrappaAttribute(GRAPPA_SELECTION_STYLE_ATTR,"lineColor(red),lineWidth(3)");
	setGrappaAttribute(GRAPPA_DELETION_STYLE_ATTR,"lineColor(grey85),lineWidth(3),dotted");
	setGrappaAttribute(GRAPPA_FONTSIZE_ADJUSTMENT_ATTR,"0");

    }

    /**
     * Creates a directed graph that is not strict
     * A convenience method equivalent to Graph(graphName,true,false).
     *
     * @param graphName the name of this graph.
     * @see Graph#Graph(java.lang.String, boolean, boolean)
     */
    public Graph(String graphName) {
	this(graphName,true,false);
    }

    // graph initialization steps
    private void initialize(String graphName) {
	eid = nid = gid = 0;
	clearBBox();

	if(id2element != null) {
	    id2element.clear();
	}

	setGraph(this);
	setSubgraph(null);
	setIdKey(Grappa.SUBGRAPH);
	addIdMapping(this);
	setName(graphName);

	Attribute attr = null;
	Enumeration enm = getGlobalAttributePairs(Grappa.NODE);
	while(enm.hasMoreElements()) {
	    setNodeAttribute((Attribute)enm.nextElement());
	}
	enm = getGlobalAttributePairs(Grappa.EDGE);
	while(enm.hasMoreElements()) {
	    setEdgeAttribute((Attribute)enm.nextElement());
	}
	enm = getGlobalAttributePairs(Grappa.SUBGRAPH);
	while(enm.hasMoreElements()) {
	    setAttribute((Attribute)enm.nextElement());
	}

	setDelete(false);
    }

    private void setDirection(boolean directed) {
	this.directed = directed;
	if(directed) {
	    setEdgeAttribute(DIR_ATTR, "forward");
	} else {
	    setEdgeAttribute(DIR_ATTR, "none");
	}
    }

    /**
     * Sets or unsets indication that paint requests should be done
     * within a synchronized wrapper that prevents concurrent paints
     * and any paints between calls to the dropcloth method.
     *
     * @param sync value to which indicator will be set
     * @return the previous indicator value
     * @see Graph#dropcloth(boolean, boolean)
     */
    public boolean setSynchronizePaint(boolean sync) {
	boolean oldSync = synchronizePaint;
	synchronizePaint = sync;
	return(oldSync);
    }

    /**
     * Get the current paint synchronization indicator value.
     *
     * @return the current paint synchronization indicator value
     * @see Graph#setSynchronizePaint(boolean)
     */
    public boolean getSynchronizePaint() {
	return(synchronizePaint);
    }

    /**
     * Sets and unsets a flag in a synchronized manner so that during the
     * period that the flag is set, painting will not occur.
     *
     * @param block value to which to set the indicator flag
     * @param auto when block is false, setting this parameter true will request a repaint() if any paint requests arrived while the dropcloth was laid out.
     * @return returns false only when block is true and a paint is pending or in progress.
     * @see Graph#setSynchronizePaint(boolean)
     *
     */
    public boolean dropcloth(boolean block, boolean auto) {
	return setBlocked(block, false, auto);
    }

    // used in GrappaPanel 
    boolean setPaint(boolean paint) {
	return setBlocked(paint, true, false);
    }

    private synchronized boolean setBlocked(boolean state, boolean isPaint, boolean repaint) {
	if(isPaint) {
	    if(state) {
		paintCalled = true;
		if(busy) {
		    return(false);
		} else {
		    return(busy = true);
		}
	    } else {
		paintCalled = busy = false;
		return(true);
	    }
	} else if(state) {
	    if(paintCalled) return(false);
	    return(busy = true);
	} else {
	    if(!paintCalled) {
		busy = false;
	    } else {
		busy = paintCalled = false;
		if(repaint) repaint();
	    }
	    return(true);
	}
    }

    /**
     * Gets Grappa default attribute.
     *
     * @param key the search key for the corresponding attribute.
     * @exception IllegalArgumentException whenever the key is null
     * @return the value of the matching Grappa default attribute or null.
     */
    public Attribute getGrappaAttribute(String key) throws IllegalArgumentException {
	if(key == null) {
	    throw new IllegalArgumentException("key value cannot be null");
	}
	if(grattributes == null) return null;
	return ((Attribute)(grattributes.get(key)));
    }

    /**
     * Gets Grappa default attribute value.
     *
     * @param key the search key for the corresponding attribute.
     * @exception IllegalArgumentException whenever the key is null
     * @return the value portion of the matching Grappa default attribute or null.
     */
    public Object getGrappaAttributeValue(String key) throws IllegalArgumentException {
	if(key == null) {
	    throw new IllegalArgumentException("key value cannot be null");
	}
	if(grattributes == null) return null;
	Attribute attr =  (Attribute)(grattributes.get(key));
	if(attr == null) return null;
	return(attr.getValue());
    }

    /**
     * Sets a Grappa package attribute.  A Grappa package attribute is one
     * specific to Grappa (for example, a display color) rather than an
     * attribute that relates to a graph.
     *
     * @param key the search key for the corresponding attribute.
     * @exception IllegalArgumentException whenever the key is not prefixed by Grappa.PKG_LOWER
     * @return the previous value of the matching Grappa default attribute or null.
     * @see GrappaConstants#PKG_LOWER
     */
    public Object setGrappaAttribute(String key, String value) throws IllegalArgumentException {
	if(grattributes == null) {
	    grattributes = new Hashtable(4);
	}
	// the get also tests if key is null
	Attribute oldValue = getGrappaAttribute(key);
	if(oldValue == null) {
	    if(!validGrappaAttributeKey(key)) {
		throw new IllegalArgumentException(Grappa.PKG_UPLOW + " attribute key must use \"" + Grappa.PKG_LOWER + "\" as a prefix");
	    }
	    oldValue = new Attribute(SYSTEM,key,value);
	    grattributes.put(key,oldValue);
	    return null;
	} else {
	    grattributes.put(key, new Attribute(SYSTEM,key,value));
	}
	return oldValue.getValue();
    }

    /**
     * Returns the attribute conversion type for the supplied attribute name.
     * Only graph global specific attribute name/type mappings are checked.
     *
     * @param attrname the attribute name
     * @return the currently associated attribute type
     */
    public static int attributeType(String attrname) {
	int convtype = -1;
	int hashCode;

	if(attrname != null) {
	    hashCode = attrname.hashCode();

	    if(hashCode == GRAPPA_BACKGROUND_COLOR_HASH && attrname.equals(GRAPPA_BACKGROUND_COLOR_ATTR)) {
		convtype = COLOR_TYPE;
	    } else if(hashCode == GRAPPA_SELECTION_STYLE_HASH && attrname.equals(GRAPPA_SELECTION_STYLE_ATTR)) {
		convtype = STYLE_TYPE;
	    } else if(hashCode == GRAPPA_DELETION_STYLE_HASH && attrname.equals(GRAPPA_DELETION_STYLE_ATTR)) {
		convtype = STYLE_TYPE;
	    } else if(hashCode == GRAPPA_FONTSIZE_ADJUSTMENT_HASH && attrname.equals(GRAPPA_FONTSIZE_ADJUSTMENT_ATTR)) {
		convtype = INTEGER_TYPE;
	    } else {
		convtype = STRING_TYPE;
	    }
	}

	return(convtype);
    }

    /**
     * Get an enumeration of the Grappa package attribute keys.
     *
     * @return an Enumeration of Attribute objects
     */
    public Enumeration getGrappaAttributeKeys() {
	if(grattributes == null) {
	    return Grappa.emptyEnumeration.elements();
	}
	return grattributes.keys();
    }

    /**
     * Check if the given key has a format consistent with Grappa package
     * attribute keys.  A Grappa package key starts with Grappa.PKG_LOWER.
     *
     * @param key the key to validate
     * @return true if the supplied key could serve as a Grappa package attribute key.
     * @see GrappaConstants#PKG_LOWER
     */
    public static boolean validGrappaAttributeKey(String key) {
	return (key != null && key.startsWith(Grappa.PKG_LOWER) && key.length() > Grappa.PKG_LOWER.length());
    }

    /**
     * Gets a graph default attribute. A graph default attribute determines
     * basic graph characteristics initially (e.g., node shape).
     *
     * @param type indicates attribute type.
     * @param key the search key for the corresponding attribute.
     * @exception IllegalArgumentException whenever the specified type is not valid
     * @return the value of the matching graph default attribute or null.
     * @see GrappaConstants#NODE
     * @see GrappaConstants#EDGE
     * @see GrappaConstants#SUBGRAPH
     */
    public static Attribute getGlobalAttribute(int type, String key) throws IllegalArgumentException {
	switch(type) {
	case Grappa.NODE:
	    return((Attribute)sysdfltNodeAttributes.get(key));
	case Grappa.EDGE:
	    return((Attribute)sysdfltEdgeAttributes.get(key));
	case Grappa.SUBGRAPH:
	    return((Attribute)sysdfltGraphAttributes.get(key));
	}
	throw new IllegalArgumentException("specified type must be NODE, EDGE or SUBGRAPH");
    }

    /**
     * Gets an enumeration of the specified graph default attribute keys
     *
     * @param type indicates attribute type.
     * @exception IllegalArgumentException whenever the specified type is not valid
     * @return an Enumeration of String objects
     * @see GrappaConstants#NODE
     * @see GrappaConstants#EDGE
     * @see GrappaConstants#SUBGRAPH
     */
    public static Enumeration getGlobalAttributeKeys(int type) throws IllegalArgumentException {
	switch(type) {
	case Grappa.NODE:
	    return(sysdfltNodeAttributes.keys());
	case Grappa.EDGE:
	    return(sysdfltEdgeAttributes.keys());
	case Grappa.SUBGRAPH:
	    return(sysdfltGraphAttributes.keys());
	}
	throw new IllegalArgumentException("specified type must be NODE, EDGE or SUBGRAPH");
    }

    /**
     * Gets an enumeration of the specified graph default attributes
     *
     * @param type indicates attribute type.
     * @exception IllegalArgumentException whenever the specified type is not valid
     * @return an Enumeration of Attribute objects
     * @see GrappaConstants#NODE
     * @see GrappaConstants#EDGE
     * @see GrappaConstants#SUBGRAPH
     */
    public static Enumeration getGlobalAttributePairs(int type) throws IllegalArgumentException {
	switch(type) {
	case Grappa.NODE:
	    return(sysdfltNodeAttributes.elements());
	case Grappa.EDGE:
	    return(sysdfltEdgeAttributes.elements());
	case Grappa.SUBGRAPH:
	    return(sysdfltGraphAttributes.elements());
	}
	throw new IllegalArgumentException("specified type must be NODE, EDGE or SUBGRAPH");
    }

    /**
     * Get a count of the graph default attributes of a particular type.
     *
     * @param type indicates attribute type.
     * @exception IllegalArgumentException whenever the specified type is not valid
     * @return a count of the specified graph default attributes
     * @see GrappaConstants#NODE
     * @see GrappaConstants#EDGE
     * @see GrappaConstants#SUBGRAPH
     */
    public static int getGlobalAttributeSize(int type) throws IllegalArgumentException {
	switch(type) {
	case Grappa.NODE:
	    return(sysdfltNodeAttributes.size());
	case Grappa.EDGE:
	    return(sysdfltEdgeAttributes.size());
	case Grappa.SUBGRAPH:
	    return(sysdfltGraphAttributes.size());
	}
	throw new IllegalArgumentException("specified type must be NODE, EDGE or SUBGRAPH");
    }

    /**
     * Add id to element lookup table
     * (used in setId method)
     *
     * @param elem the element associated with the id
     */
    Element addIdMapping(Element elem) {
	if(elem == null) {
	    return null;
	}
	if(id2element == null) {
	    id2element = new Hashtable();
	}
	return (Element)id2element.put(elem.getIdKey(),elem);
    }

    /**
     * Creates a id key given a type and id number.
     *
     * @param type one of Grappa.NODE, Grappa.EDGE or Grappa.SUBGRAPH
     * @param id an id number
     * @exception IllegalArgumentException whenever the specified type is not valid
     * @return an idKey for an element
     * @see GrappaConstants#NODE
     * @see GrappaConstants#EDGE
     * @see GrappaConstants#SUBGRAPH
     */
    static Long idMapKey(int type, int id) throws IllegalArgumentException {
	long value = (long)(id);
	int tval = (type&(Grappa.NODE|Grappa.EDGE|Grappa.SUBGRAPH));
	if(tval == 0) {
	    throw new IllegalArgumentException("supplied type does not specify node, edge or subgraph");
	}
	value = (value << Grappa.TYPES_SHIFT) | (type&(Grappa.NODE|Grappa.EDGE|Grappa.SUBGRAPH));
	return new Long(value);
    }

    /**
     * Get the type of the id key.
     *
     * @param idKey the id key to examine
     * @return the type of the id key (Grappa.NODE, Grappa.EDGE, Grappa.SUBGRAPH)
     * @see GrappaConstants#NODE
     * @see GrappaConstants#EDGE
     * @see GrappaConstants#SUBGRAPH
     */
    static int idKeyType(Long idKey) {
	long value = idKey.longValue();
	return (int)(value&(Grappa.NODE|Grappa.EDGE|Grappa.SUBGRAPH));
    }

    /**
     * Get the type of the id key.
     *
     * @param idKey the id key to examine
     * @return the type of the id key (Grappa.NODE, Grappa.EDGE, Grappa.SUBGRAPH)
     * @see GrappaConstants#NODE
     * @see GrappaConstants#EDGE
     * @see GrappaConstants#SUBGRAPH
     */
    static int idKeyId(Long idKey) {
	long value = idKey.longValue();
	return (int)(value>>>Grappa.TYPES_SHIFT);
    }

    /**
     * Get the element associated with an id key
     *
     * @param idKey the id key of the element to be located
     * @return the Element object matching the id key or null
     */
    Element element4Id(Long idKey) {
	if(id2element == null) {
	    return null;
	}
	return (Element)id2element.get(idKey);
    }

    /**
     * Remove id2element dictionary element
     *
     * @param id the id number of the element entry to be removed
     */
    void removeIdMapping(Element elem) {
	if(id2element != null && elem != null) {
	    id2element.remove(elem.getIdKey());
	}
    }

    /**
     * Output graph to specified Writer.
     *
     * @param output the Writer for writing
     */
    public void printGraph(Writer output) {
	PrintWriter out = null;
    
	if(output instanceof PrintWriter) {
	    out = (PrintWriter)output;
	} else {
	    out = new PrintWriter(output);
	}
	getGraph().printSubgraph(out);
	out.flush();
    }

    /**
     * Output graph to specified OutputStream.
     * A convenience method to accomodate the OuputStreams easily.
     *
     * @param output the OutputStream for writing
     */
    public void printGraph(OutputStream output) {
	printGraph(new PrintWriter(output));
    }

    /**
     * Get the next id number for the specified type and increment the counter.
     *
     * @param type type of id number to return
     * @exception IllegalArgumentException whenever the specified type is not valid
     * @return the next sequential id number (counter is incremented).
     * @see GrappaConstants#NODE
     * @see GrappaConstants#EDGE
     * @see GrappaConstants#SUBGRAPH
     */
    int nextId(int type) throws IllegalArgumentException {
	switch(type) {
	case Grappa.NODE:
	    return(nid++);
	case Grappa.EDGE:
	    return(eid++);
	case Grappa.SUBGRAPH:
	    return(gid++);
	}
	throw new IllegalArgumentException("Type ("+type+") is not recognized.");
    }

    /**
     * Get the next id number for the specified type, but do not increment the counter.
     *
     * @param type type of id number to return
     * @exception IllegalArgumentException whenever the specified type is not valid
     * @return the next sequential id number (counter is not incremented).
     * @see GrappaConstants#NODE
     * @see GrappaConstants#EDGE
     * @see GrappaConstants#SUBGRAPH
     */
    public int getId(int type) throws IllegalArgumentException {
	switch(type) {
	case Grappa.NODE:
	    return(nid);
	case Grappa.EDGE:
	    return(eid);
	case Grappa.SUBGRAPH:
	    return(gid);
	}
	throw new IllegalArgumentException("Type ("+type+") is not recognized.");
    }

    /**
     * Get the current indent string.
     *
     * @return the current indent string.
     */
    public String getIndent() {
	if(indent == null) {
	    indent = new StringBuffer(5 * INDENT_STRING.length());
	}
	return(indent.toString());
    }

    /**
     * Increase the indent string by appending INDENT_STRING.
     *
     * @see Graph#INDENT_STRING
     */
    public void incrementIndent() {
	if(indent == null) {
	    indent = new StringBuffer(5 * INDENT_STRING.length());
	}
	indent.append(INDENT_STRING);
    }

    /**
     * Decrease the indent string by removing one INDENT_STRING.
     *
     * @see Graph#INDENT_STRING
     */
    public void decrementIndent() {
	int len = indent.length();

	if(len == 0) return;

	if(len < INDENT_STRING.length()) {
	    indent.setLength(0);
	} else {
	    indent.setLength(len - INDENT_STRING.length());
	}
    }

    /**
     * Check if the graph is directed.
     *
     * @return true if graph is a directed graph
     */
    public boolean isDirected() {
	return(directed);
    }

    /**
     * Check if the graph is strict (i.e., no self-loops).
     *
     * @return true if the graph is strict
     */
    public boolean isStrict() {
	return(strict);
    }

    /**
     * Set the tooltip text displayed when outside the graph area.
     *
     * @param text out-of-graph tooltip text
     * @return previous out-of-graph tooltip text
     */
    public String setToolTipText(String text) {
	String oldTip = toolTipText;
	toolTipText = text;
	return(oldTip);
    }

    /**
     * Get the tooltip text displayed when outside the graph area.
     *
     * @return out-of-graph tooltip text
     */
    public String getToolTipText() {
	return(toolTipText);
    }

    //TODO find out dot options to fill or outline graph also orientation.

    /**
     * Reset this graph by removing all its elements and re-initiailizing
     * its internal variables.
     */
    public void reset() {
	String graphName = getName();
	if(delete()) initialize(graphName);
    }

    /**
     * Reset this graph by removing all its elements and re-initiailizing
     * its internal variables and possibly changing its name, directedness
     * and strictness.
     */
    public void reset(String graphName, boolean directed, boolean strict) {
	name = graphName;
	reset();
	setDirection(directed);
	this.strict = strict;
    }

    /**
     * Check if this graph is interactively editable (i.e., through mouse events).
     *
     * @return true if the graph can be edited interactively.
     */
    public boolean isEditable() {
	return editable;
    }

    /**
     * Set the editability of the graph.
     *
     * @param mode true to turn on editability.
     * @return previous value
     * @see Graph#isEditable()
     */
    public boolean setEditable(boolean mode) {
	boolean wasMode = editable;
	editable = mode;
	return wasMode;
    }

    /**
     * Check if graph elements are interactively selectable  (i.e., through mouse events).
     *
     * @return true if graph elements can be selected interactively.
     */
    public boolean isSelectable() {
	return selectable;
    }

    /**
     * Set the selectability of the graph.
     *
     * @param mode true to turn on selectability.
     * @return previous value
     * @see Graph#isSelectable()
     */
    public boolean setSelectable(boolean mode) {
	boolean wasMode = selectable;
	selectable = mode;
	return wasMode;
    }

    /**
     * Check if an element-specific menu is available interactively (i.e., through mouse events).
     *
     * @return true if an element-specific menu is available
     */
    public boolean isMenuable() {
	return menuable;
    }

    /**
     * Set whether element-specific menus are to be available interactively.
     *
     * @param mode true to turn on element-specific-menus.
     * @return previous value
     * @see Graph#isMenuable()
     */
    public boolean setMenuable(boolean mode) {
	boolean wasMode = menuable;
	menuable = mode;
	return wasMode;
    }

    /**
     * Set the PrintWriter for error messages.
     *
     * @param errWriter the PrintWriter to use for error messages.
     * @return the previous PrintWriter used for error messages.
     * @see java.io.PrintWriter
     */
    public PrintWriter setErrorWriter(PrintWriter errWriter) {
	PrintWriter oldWriter = this.errWriter;
	this.errWriter = errWriter;
	return oldWriter;
    }

    /**
     * Get the current PrintWriter used for error messages.
     *
     * @return the current PrintWriter used for error messages.
     * @see java.io.PrintWriter
     */
    public PrintWriter getErrorWriter() {
	return errWriter;
    }

    /**
     * Print the supplied message to the error output.
     * Nothing happens if the error output is set to null.
     *
     * @param msg the message to print on the error output.
     * @see Graph#setErrorWriter(java.io.PrintWriter)
     */
    public void printError(String msg) {
	printError(msg,null);
    }

    /**
     * Print the supplied message and exception information to the error output.
     * Nothing happens if the error output is set to null.
     *
     * @param msg the message to print on the error output.
     * @param ex if supplied, the stack trace associated with this exception is also printed.
     * @see Graph#setErrorWriter(java.io.PrintWriter)
     */
    public void printError(String msg, Exception ex) {
	if(getErrorWriter() == null) return;
	getErrorWriter().println("ERROR: " + msg);
	if(ex != null) ex.printStackTrace(getErrorWriter());
	getErrorWriter().flush();
    }

    //////////////////////////////////////////////////////////////////////

    /**
     * Builds any GrappaNexus object not already built for elements in
     * this graph.
     */
    public void buildShapes() {
	GraphEnumeration enm = elements();
	Element elem;

	while(enm.hasMoreElements()) {
	    elem = enm.nextGraphElement();
	    if(elem.grappaNexus == null) {
		elem.buildShape();
	    }
	}
    }

    /**
     * Builds any GrappaNexus object not already built and rebuilds those
     * that already exist for all elements in this graph.
     */
    public void resync() {
	Element elem = null;
	GraphEnumeration enm = elements();
	while(enm.hasMoreElements()) {
	    elem = enm.nextGraphElement();
	    if(elem.grappaNexus == null) elem.buildShape();
	    else elem.grappaNexus.rebuild();
	}
    }

    //////////////////////////////////////////////////////////////////////

    /**
     * Makes a repaint request of all GrappaPanels that are displaying
     * this graph.
     */
    public void repaint() {
	if(panelList == null) return;

	boolean incomplete = true;

	ListIterator li = null;

	while(incomplete) {
	    try {
		li = panelList.listIterator(0);
		while(li.hasNext()) {
		    ((GrappaPanel)li.next()).repaint();
		}
	    } catch(ConcurrentModificationException cme) {
		continue;
	    }
	    incomplete = false;
	}
    }

    /**
     * Makes a paintImmediately request of all GrappaPanels that are displaying
     * this graph.
     */
    public void paintImmediately() {
	if(panelList == null) return;

	boolean incomplete = true;

	ListIterator li = null;

	GrappaPanel panel = null;

	while(incomplete) {
	    try {
		li = panelList.listIterator(0);
		while(li.hasNext()) {
		    panel = ((GrappaPanel)li.next());
		    panel.paintImmediately(panel.getVisibleRect());
		}
	    } catch(ConcurrentModificationException cme) {
		continue;
	    }
	    incomplete = false;
	}
    }

    /**
     * Adds a panel to the list of GrappaPanels that are displaying
     * this graph.
     *
     * @param panel the GrappaPanel to be added to the list
     */
    public void addPanel(GrappaPanel panel) {
	if(panelList == null) {
	    panelList = Collections.synchronizedList(new LinkedList());
	}
	synchronized(panelList) {
	    if(!panelList.contains(panel)) panelList.add(panel);
	}
    }

    /**
     * Removes a panel to the list of GrappaPanels that are displaying
     * this graph.
     *
     * @param panel the GrappaPanel to be removed to the list
     */
    public void removePanel(GrappaPanel panel) {
	if(panelList == null) return;
	synchronized(panelList) {
	    panelList.remove(panel);
	}
    }
}

