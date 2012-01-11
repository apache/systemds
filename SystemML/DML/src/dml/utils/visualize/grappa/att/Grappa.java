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

/**
 * This class sets default option variables and other set-up.
 * In addition, some convenience methods for exception display are
 * included and some lookup tables are initialized.
 *
 * <P>The Grappa package itself has two roles:<UL>
 * <LI>building, maintaining and manipulating graph structure, and
 * <LI>drawing a positioned graph.</UL>
 * Grappa itself does not have any methods for graph positioning.
 * Grappa simply draws the nodes and edges based on the value of the
 * element's <TT>pos</TT> attribute, though it will treat an unpositioned
 * edge as a straight line between the center points of its attached nodes.
 *
 * Some graph layout references are:<OL>
 * <LI>"An Algorithm for Drawing General Undirected Graphs",
 * Tomihisa Kamada and Satoru Kawai,
 * <I>Information Processing Letters</I>, Vol. 31 (1989), pp. 7 - 15
 *
 * <LI>"Graph Drawing by Force-directed Placement",
 * Thomas M. J. Fruchterman and Edward M. Reingold,
 * <I>Software - Practice and Experience</I>, Vol. 21 No. 11 (1991), pp. 1129 - 1164
 *
 * <LI>"A Technique for Drawing Directed Graphs",
 * Emden R. Gansner, Eleftherios Koutsofios, Stephen C. North and
 * Kiem-Phong Vo,
 * <I>IEEE Transactions on Software Engineering</I>, Vol 19 No. 3 (1993), pp. 214 - 230
 *
 * <LI>"NicheWorks - Interactive Visualization of Very Large Graphs",
 * Graham J. Wills,
 * <A HREF="http://www.bell-labs.com:80/user/gwills/NICHEguide/nichepaper.html" TARGET="_blank">http://www.bell-labs.com:80/user/gwills/NICHEguide/nichepaper.html</A>, circa 1997</OL>
 *
 * <P>Grappa does supply a utility, GrappaSupport.filterGraph(), for updating
 * a graph (including position information) from I/O streams such as might
 * be obtained from a java.net.URLConnection.
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public abstract class Grappa
    implements dml.utils.visualize.grappa.att.GrappaConstants
{

    /**
     * Look-up table that maps a shape name to its integer reference value.
     */
    public static java.util.Hashtable keyToShape = new java.util.Hashtable();
    /**
     * Look-up table that maps a shape reference value to its name.
     */
    public static java.util.Hashtable shapeToKey = new java.util.Hashtable();

    /*
     * Set-up lookup table that defines recognized shapes (useful
     * for switch statements).
     */
    static {
	keyToShape.put("box", new Integer(BOX_SHAPE));
	keyToShape.put("circle", new Integer(OVAL_SHAPE));
	keyToShape.put("custom", new Integer(CUSTOM_SHAPE));
	keyToShape.put("diamond", new Integer(DIAMOND_SHAPE));
	keyToShape.put("doublecircle", new Integer(DOUBLECIRCLE_SHAPE));
	keyToShape.put("doubleoctagon", new Integer(DOUBLEOCTAGON_SHAPE));
	keyToShape.put("egg", new Integer(EGG_SHAPE));
	keyToShape.put("ellipse", new Integer(OVAL_SHAPE));
	keyToShape.put("hexagon", new Integer(HEXAGON_SHAPE));
	keyToShape.put("house", new Integer(HOUSE_SHAPE));
	keyToShape.put("invhouse", new Integer(INVERTEDHOUSE_SHAPE));
	keyToShape.put("invtrapezium", new Integer(INVERTEDTRAPEZIUM_SHAPE));
	keyToShape.put("invtriangle", new Integer(INVERTEDTRIANGLE_SHAPE));
	keyToShape.put("octagon", new Integer(OCTAGON_SHAPE));
	keyToShape.put("pentagon", new Integer(PENTAGON_SHAPE));
	keyToShape.put("parallelogram", new Integer(PARALLELOGRAM_SHAPE));
	keyToShape.put("plaintext", new Integer(PLAINTEXT_SHAPE));
	keyToShape.put("point", new Integer(POINT_SHAPE));
	keyToShape.put("polygon", new Integer(POLYGON_SHAPE));
	keyToShape.put("record", new Integer(RECORD_SHAPE));
	keyToShape.put("roundedbox", new Integer(ROUNDEDBOX_SHAPE));
	keyToShape.put("trapezium", new Integer(TRAPEZIUM_SHAPE));
	keyToShape.put("triangle", new Integer(TRIANGLE_SHAPE));
	keyToShape.put("tripleoctagon", new Integer(TRIPLEOCTAGON_SHAPE));
	keyToShape.put("Mcircle", new Integer(MCIRCLE_SHAPE));
	keyToShape.put("Mdiamond", new Integer(MDIAMOND_SHAPE));
	keyToShape.put("Mrecord", new Integer(MRECORD_SHAPE));
	keyToShape.put("Msquare", new Integer(MSQUARE_SHAPE));

	java.util.Enumeration enm = keyToShape.keys();
	String key = null;
	while(enm.hasMoreElements()) {
	    key = (String)enm.nextElement();
	    shapeToKey.put(keyToShape.get(key),key);
	}

	// special case, but don't want it in shapeToKey
	keyToShape.put("square", new Integer(BOX_SHAPE));
    }

    /**
     * The java.awt.Toolkit.getDefaultToolkit() value, if available.
     * It is not populated until required, so if not graph display
     * is requested, it will remain null.
     */
    public static java.awt.Toolkit toolkit = null;

  
    /*
     * A instance of ExceptionDisplay used with the (public) convenience
     * methods that follow.
     */ 
    private static final ExceptionDisplay exceptionDisplay = new ExceptionDisplay(Grappa.PKG_UPLOW + ":  Exception Detected");

    /**
     * Boolean for enabling/disabling exception pop-up window display.
     */
    public static boolean doDisplayException = true;

    /**
     * Method for displaying an exception in a pop-up window (if enabled).
     * @param ex The exception value about which information is to be displayed.
     * @see Grappa#doDisplayException
     * @see DisplayException
     */
    public static void displayException(java.lang.Exception ex) {
	if(doDisplayException) exceptionDisplay.displayException(ex);
    }

    /**
     * Method for displaying an exception in a pop-up window (if enabled).
     * @param ex The exception value about which information is to be displayed.
     * @param msg Additional text to be displayed ahead of exception info.
     * @see Grappa#doDisplayException
     * @see DisplayException
     */
    public static void displayException(java.lang.Exception ex, java.lang.String msg) {
	if(doDisplayException) exceptionDisplay.displayException(ex,msg);
    }

    /**
     * A convenience Vector useful when an enumeration is to be returned, but
     * the object to be enumerated is null (in which case, the return value can
     * be <I>Grappa.emptyEnumeration.elements()</I>, whose <I>hasMoreElements()</I> method
     * will return <B>false</B>).
     */
    public static final java.util.Vector emptyEnumeration = new java.util.Vector(0,0);

    /*
     * Default tool-tip text when cursor is outside graph, but inside
     * the display panel.
     */
    private static String toolTipText = "<outside graph boundary>";

    /**
     * Sets the tool-tip text displayed when outside graph, but inside
     * the display panel.
     *
     * @param text The new outside-the-graph tool-tip text to display.
     *
     * @return The former outside-the-graph tool-tip text.
     */
    public static String setToolTipText(String text) {
	String oldTip = toolTipText;
	toolTipText = text;
	return(oldTip);
    }

    /**
     * Gets the current  tool-tip text displayed when outside graph, but inside
     * the display panel.
     *
     * @return The current outside-the-graph tool-tip text.
     */
    public static String getToolTipText() {
	return(toolTipText);
    }

    /*
     * global options
     */

    /**
     * Boolean to indicate if all element attributes should be printed.
     * By default, only the element specific attributes are printed and
     * attributes inherited from a parent element (subgraph) are skipped.
     *
     * @see Subgraph#printSubgraph
     * @see Element#printAllAttributes
     */
    public static boolean elementPrintAllAttributes	= false;
    /**
     * Boolean to indicate if the default attributes associated with
     * a subgraph should be printed.
     * By default, only those differing from the pre-defined defaults are
     * printed.
     *
     * @see Subgraph#printSubgraph
     * @see Element#printDefaultAttributes
     */
    public static boolean elementPrintDefaultAttributes	= false;

    /**
     * Indicates if element text should be included in the element
     * bounding box. By default, it is.
     *
     * @see GrappaNexus#boundText
     */
    public static boolean shapeBoundText		= true;
    /**
     * Indicates if the area bounding the element text should be
     * filled/outlined along with the element when being drawn.
     * By default, it is not.
     *
     * @see GrappaNexus#clearText
     */
    public static boolean shapeClearText		= false;
    /**
     * Indicates if element text should be drawn when drawing the
     * element. By default, it is.
     *
     * @see GrappaNexus#drawText
     */
    public static boolean shapeDrawText			= true;

    /**
     * Indicates if element node position indicates the node center
     * point. Otherwise, the position is assumed to indicate the
     * upper-left corner of the nodes bounding box.
     * By default, the position indicates the node's center point.
     */
    public static boolean centerPointNodes		= true;
    /**
     * Indicates if the label position for node labels should automatically
     * be set to the center point of the node. When true, positioning
     * information provided by the <TT>lp</TT> attribute is completely
     * ignored. By default, auto-positioning is used.
     */
    public static boolean autoPositionNodeLabel		= true;

    // not used
    //public static boolean graphAutoUpdate		= false;

    /**
     * Indicates if the <TT>bb</TT> attribute of a subgraph should
     * automatically be set whenever the bounding box is calculated.
     * By default, the attribute is not set automatically.
     */
    public static boolean provideBBoxAttribute		= false;

    /**
     * Indicates what winding rule to use whenever a winding rule is
     * required. The default is the java.awt.geom.PathIterator
     * WIND_NON_ZERO rule.
     */
    public static int windingRule			= java.awt.geom.PathIterator.WIND_NON_ZERO;

    /**
     * Indicates whether the <TT>orientation</TT> attribute is specifed
     * in degrees rather than radians. Degrees is the default.
     */
    public static boolean orientationInDegrees          = true;
    /**
     * Indicates whether the <TT>rotation</TT> attribute is specifed
     * in degrees rather than radians. Degrees is the default.
     */
    public static boolean rotationInDegrees  	        = true;

    /**
     * Indicates whether only the list of attributes found in the
     * PRINTLIST_ATTR should be printed. See also printVisibleOnly.
     * The default is false.
     */
    public static boolean usePrintList        		= false;

    /**
     * Indicates whether only visible elements should be included
     * when printing a graph.
     * The default is false.
     */
    public static boolean printVisibleOnly    		= false;

    /**
     * Indicates whether anti-aliasing should be used when drawing.
     * The default is true.
     */
    public static boolean useAntiAliasing     		= true;

    /**
     * Indicates whether anti-aliasing should be used when drawing text.
     * The default is false.
     */
    public static boolean antiAliasText			= false;
    /**
     * Indicates whether fractional metrics should be used when drawing text.
     * The default is false.
     */
    public static boolean useFractionalMetrics		= false;

    /**
     * Indicates whether the value of y-coordinates should be negated
     * when reading or writing y-coord information as string attributes.
     * Note: this indicator should be set to <I>true</I> when working
     * with string attributes generated by or to be read by the <TT>dot</TT>
     * graph layout program or to be compatible to earlier versions of Grappa.
     * The default is true.
     */
    public static boolean negateStringYCoord     	= true;

    /**
     * Indicates that graph labels, when not explicitly positioned via
     * the <TT>lp</TT> attribute, should be placed at the bottom of the
     * graph instead of the top. The default is true, meaning the label
     * will be placed at the bottom of the graph.
     */
    public static boolean labelGraphBottom      	= true;
    /**
     * Indicates that graph labels, when not explicitly positioned via
     * the <TT>lp</TT> attribute, should be placed just outside the
     * graph bounding box instead of just inside. The default is true,
     * meaning the label will be placed outside the bounding box.
     */
    public static boolean labelGraphOutside     	= true;

    /**
     * Indicates that background drawing, if any is provided via
     * a <TT>GrappaBacker</TT> implementation, should be displayed
     * or not. The default of true means the background drawing should
     * be displayed, if provided.
     */
    public static boolean backgroundDrawing     	= true;

    /**
     * When the transform scale applied when drawing in a GrappaPanel is
     * less than this value, then node labels are suppressed.
     * 
     */
    public static double nodeLabelsScaleCutoff		= 0.5;

    /**
     * Cluster subgraphs will have their bounding box outlined. To
     * similarly outline all types of subgraphs (except the root subgraph),
     * set this value to true.
     * 
     */
    public static boolean outlineSubgraphs		= false;

    /**
     * When the transform scale applied when drawing in a GrappaPanel is
     * less than this value, then edge labels are suppressed.
     * 
     */
    public static double edgeLabelsScaleCutoff		= 0.5;

    /**
     * When the transform scale applied when drawing in a GrappaPanel is
     * less than this value, then subgraph labels are suppressed.
     * 
     */
    public static double subgLabelsScaleCutoff		= 0.3;

    /**
     * Indicates whether paints should be done within a synchronized
     * wrapper. When enable the Graph dropcloth method can be used to
     * prevent paints during certain critical operations.
     *
     * @see Graph#dropcloth(boolan, boolean)
     */
    public static boolean synchronizePaint     		= false;

    /**
     * Indicates that an image requested via the IMAGE_ATTR of
     * an element should be loaded before the element is drawn.
     * By default, Grappa will wait.
     *
     * @see GrappaNexus#drawImage
     */
    public static boolean waitForImages			= true;

    /**
     * Indicates which classes of elements are suitable for selection
     * based on cursor position. Value is the logical OR of NODE,
     * EDGE and SUBGRAPH. The default is SUBGRAPH|NODE|EDGE.
     *
     */
    public static int elementSelection			= SUBGRAPH|NODE|EDGE;

}
