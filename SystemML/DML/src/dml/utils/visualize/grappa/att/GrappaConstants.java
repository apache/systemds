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
 * This class provides a common set of constant, class variables
 * used by the classes in the grappa package.
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public interface GrappaConstants
{
    /**
     * Package prefix string ("dml.hops.visualize.grappa.att.") if anyone needs it.
     */
    public final static String PACKAGE_PREFIX = "dml.hops.visualize.grappa.att.";

    /**
     * Package name as an up-low string.
     */
    public final static String PKG_UPLOW = "Grappa";
    /**
     * Package name as an upper-case string (as a convenience).
     */
    public final static String PKG_UPPER = "GRAPPA";
    /**
     * Package name as an lower-case string (as a convenience).
     */
    public final static String PKG_LOWER = "grappa";

    /**
     * The new-line string for this system (as specified by the line.spearator property).
     */
    public final static String NEW_LINE = System.getProperty("line.separator");

    /**
     * The unicode no-break space character.
     */
    public final static char NBSP = '\u00a0';


    /**
     * The identity transform (as a convenience).
     */
    public final static java.awt.geom.AffineTransform IDENTXFRM = new java.awt.geom.AffineTransform();

    /*
     * edit operations (NOT IMPLEMENTED YET)
     public final static int EDIT_UNDO              = 0;
     public final static int EDIT_CUT               = 1;
     public final static int EDIT_PASTE             = 2;
     public final static int EDIT_COPY              = 3;
     public final static int EDIT_DELETE            = 4;
     public final static int EDIT_ADD               = 5;
    */

    /**
     * Element type value indicating a node.
     */
    public final static int NODE               = 1;
    /**
     * Element type value indicating an edge.
     */
    public final static int EDGE               = 2;
    /**
     * Element type value indicating a graph (or subgraph).
     */
    public final static int SUBGRAPH           = 4;
    /**
     * System attribute indicator.
     */
    final static int SYSTEM                    = 8;

    /**
     * Natural log of 10 (as a convenience).
     */
    final static double LOG10                  = Math.log(10);

    /**
     * Bit indicator that selection highlight is active.
     */
    public final static int SELECTION_MASK		= 1;
    /**
     * Bit indicator that deletion highlight is active.
     */
    public final static int DELETION_MASK		= 2;
    /**
     * Bit mask for highlight bits.
     */
    public final static int HIGHLIGHT_MASK		= SELECTION_MASK|DELETION_MASK;
    /**
     * Bit indicator that an element should be highlighted.
     */
    public final static int HIGHLIGHT_ON		= 4;
    /**
     * Bit indicator that an element should not be highlighted.
     */
    public final static int HIGHLIGHT_OFF		= 8;
    /**
     * Bit indicator that an element's highlight should be toggled.
     */
    public final static int HIGHLIGHT_TOGGLE		= 16;

    /**
     * Maximum number of bits needed to represet the Element types.
     * Element type is merged with the element id number (a sequentially
     * assigned number) to ensure a unique identifier (within an invocation of
     * the package).
     *
     */
    public final static int TYPES_SHIFT            = 3;

    /**
     * Points per inch (72).
     */
    public final static double PointsPerInch = 72;

    /**
     * Default gap in pixels between peripheries.
     */
    public final static int PERIPHERY_GAP = 4;

    /**
     * Name prefix for name generation of unnamed subgraphs.
     */
    public final static String ANONYMOUS_PREFIX    = "_anonymous_";

    /**
     * String name for bounding box attribute (bb).
     */
    public final static String BBOX_ATTR        = "bb";
    /**
     * String name for cluster rank attribute (clusterrank).
     */
    public final static String CLUSTERRANK_ATTR       = "clusterrank";
    /**
     * String name for color attribute (color).
     */
    public final static String COLOR_ATTR       = "color";
    /**
     * String name for custom class used to draw custom shapes (custom).
     */
    public final static String CUSTOM_ATTR       = "custom";
    /**
     * String name for direction attribute (dir).
     */
    public final static String DIR_ATTR  = "dir";
    /**
     * String name for distortion attribute (distortion).
     */
    public final static String DISTORTION_ATTR  = "distortion";
    /**
     * String name for fill color attribute (fillcolor).
     */
    public final static String FILLCOLOR_ATTR   = "fillcolor";
    /**
     * String name for fontcolor attribute (fontcolor).
     */
    public final static String FONTCOLOR_ATTR   = "fontcolor";
    /**
     * String name for fontname attribute (fontname).
     */
    public final static String FONTNAME_ATTR    = "fontname";
    /**
     * String name for fontsize attribute (fontsize).
     */
    public final static String FONTSIZE_ATTR    = "fontsize";
    /**
     * String name for fontstyle attribute (fontstyle).
     */
    public final static String FONTSTYLE_ATTR   = "fontstyle";
    /**
     * String name for background color attribute (grappaBackgroundColor).
     */
    public final static String GRAPPA_BACKGROUND_COLOR_ATTR   = Grappa.PKG_LOWER+"BackgroundColor";
    /**
     * String name for selection color attribute (grappaSelectionColor).
     */
    public final static String GRAPPA_SELECTION_STYLE_ATTR   = Grappa.PKG_LOWER+"SelectionColor";
    /**
     * String name for deletion color attribute (grappaDeletionColor).
     */
    public final static String GRAPPA_DELETION_STYLE_ATTR   = Grappa.PKG_LOWER+"DeletionColor";
    /**
     * String name for fontsize adjustment attribute (grappaFontsizeAdjustment).
     */
    public final static String GRAPPA_FONTSIZE_ADJUSTMENT_ATTR   = Grappa.PKG_LOWER+"FontsizeAdjustment";
    /**
     * String name for height attribute (height).
     */
    public final static String HEIGHT_ATTR      = "height";
    /**
     * String name for image attribute (image).
     */
    public final static String IMAGE_ATTR      = "image";
    /**
     * String name for label attribute (label).
     */
    public final static String LABEL_ATTR       = "label";
    /**
     * String name for label position attribute (lp).
     */
    public final static String LP_ATTR          = "lp";
    /**
     * String name for head label attribute (headlabel).
     */
    public final static String HEADLABEL_ATTR       = "headlabel";
    /**
     * String name for head label position attribute (head_lp).
     */
    public final static String HEADLP_ATTR          = "head_lp";
    /**
     * String name for tail label attribute (taillabel).
     */
    public final static String TAILLABEL_ATTR       = "taillabel";
    /**
     * String name for tail label position attribute (tail_lp).
     */
    public final static String TAILLP_ATTR          = "tail_lp";
    /**
     * String name for label position attribute (margin).
     */
    public final static String MARGIN_ATTR      = "margin";
    /**
     * String name for mincross limit attribute [unused] (mclimit).
     */
    public final static String MCLIMIT_ATTR      = "mclimit";
    /**
     * String name for minimum subgraph bounding box attribute (minbox).
     */
    public final static String MINBOX_ATTR      = "minbox";
    /**
     * String name for minimum rank distance between head and tail of edges attribute [unused] (minlen).
     */
    public final static String MINLEN_ATTR      = "minlen";
    /**
     * String name for minimum subgraph size attribute (minsize).
     */
    public final static String MINSIZE_ATTR      = "minsize";
    /**
     * String name for node separation attribute [unused] (nodesep).
     */
    public final static String NODESEP_ATTR     = "nodesep";
    /**
     * String name for orientation angle attribute (orientation).
     */
    public final static String ORIENTATION_ATTR = "orientation";
    /**
     * String name for patch work attribute (patch).
     */
    public final static String PATCH_ATTR         = "patch";
    /**
     * String name for peripheries attribute (peripheries).
     */
    public final static String PERIPHERIES_ATTR = "peripheries";
    /**
     * String name for position attribute (pos).
     */
    public final static String POS_ATTR         = "pos";
    /**
     * String name for print list attribute (printlist).
     */
    public final static String PRINTLIST_ATTR      = "printlist";
    /**
     * String name for rank direction attribute [unused] (rankdir).
     */
    public final static String RANKDIR_ATTR     = "rankdir";
    /**
     * String name for rank separation attribute [unused] (ranksep)
     */
    public final static String RANKSEP_ATTR      = "ranksep";
    /**
     * String name for rectangles attribute (rects).
     */
    public final static String RECTS_ATTR       = "rects";
    /**
     * String name for rotation attribute (rotation).
     */
    public final static String ROTATION_ATTR       = "rotation";
    /**
     * String name for shape attribute (shape).
     */
    public final static String SHAPE_ATTR       = "shape";
    /**
     * String name for sides attribute (sides).
     */
    public final static String SIDES_ATTR       = "sides";
    /**
     * String name for size attribute [unused] (size).
     */
    public final static String SIZE_ATTR        = "size";
    /**
     * String name for skew attribute (skew).
     */
    public final static String SKEW_ATTR        = "skew";
    /**
     * String name for style attribute (style).
     */
    public final static String STYLE_ATTR       = "style";
    /**
     * String name for tag attribute (tag).
     */
    public final static String TAG_ATTR         = "tag";
    /**
     * String name for tip attribute (tip).
     */
    public final static String TIP_ATTR         = "tip";
    /**
     * String name for weight attribute [unused] (weight).
     */
    public final static String WEIGHT_ATTR      = "weight";
    /**
     * String name for width attribute (width).
     */
    public final static String WIDTH_ATTR       = "width";

    /**
     * Hash code for bounding box attribute (bb).
     */
    public final static int BBOX_HASH        = BBOX_ATTR.hashCode();
    /**
     * Hash code for color attribute (color).
     */
    public final static int COLOR_HASH       = COLOR_ATTR.hashCode();
    /**
     * Hash code for custom attribute (custom).
     */
    public final static int CUSTOM_HASH       = CUSTOM_ATTR.hashCode();
    /**
     * Hash code for edge direction attribute (dir).
     */
    public final static int DIR_HASH       = DIR_ATTR.hashCode();
    /**
     * Hash code for distortion attribute (distortion).
     */
    public final static int DISTORTION_HASH  = DISTORTION_ATTR.hashCode();
    /**
     * Hash code for fillcolor attribute (fillcolor).
     */
    public final static int FILLCOLOR_HASH   = FILLCOLOR_ATTR.hashCode();
    /**
     * Hash code for fontcolor attribute (fontcolor).
     */
    public final static int FONTCOLOR_HASH   = FONTCOLOR_ATTR.hashCode();
    /**
     * Hash code for fontname attribute (fontname).
     */
    public final static int FONTNAME_HASH    = FONTNAME_ATTR.hashCode();
    /**
     * Hash code for fontsize attribute (fontsize).
     */
    public final static int FONTSIZE_HASH    = FONTSIZE_ATTR.hashCode();
    /**
     * Hash code for fontstyle attribute (fontstyle).
     */
    public final static int FONTSTYLE_HASH   = FONTSTYLE_ATTR.hashCode();
    /**
     * Hash code for background color attribute (grappaBackgroundColor).
     */
    public final static int GRAPPA_BACKGROUND_COLOR_HASH   = GRAPPA_BACKGROUND_COLOR_ATTR.hashCode();
    /**
     * Hash code for selection color attribute (grappaSelectionColor).
     */
    public final static int GRAPPA_SELECTION_STYLE_HASH   = GRAPPA_SELECTION_STYLE_ATTR.hashCode();
    /**
     * Hash code for deletion color attribute (grappaDeletionColor).
     */
    public final static int GRAPPA_DELETION_STYLE_HASH   = GRAPPA_DELETION_STYLE_ATTR.hashCode();
    /**
     * Hash code for fontsize adjustment attribute (grappaFontsizeAdjustment).
     */
    public final static int GRAPPA_FONTSIZE_ADJUSTMENT_HASH   = GRAPPA_FONTSIZE_ADJUSTMENT_ATTR.hashCode();
    /**
     * Hash code for height attribute (height).
     */
    public final static int HEIGHT_HASH      = HEIGHT_ATTR.hashCode();
    /**
     * Hash code for image attribute (image).
     */
    public final static int IMAGE_HASH       = IMAGE_ATTR.hashCode();
    /**
     * Hash code for label attribute (label).
     */
    public final static int LABEL_HASH       = LABEL_ATTR.hashCode();
    /**
     * Hash code for label position attribute (lp).
     */
    public final static int LP_HASH          = LP_ATTR.hashCode();
    /**
     * Hash code for head label attribute (headlabel).
     */
    public final static int HEADLABEL_HASH       = HEADLABEL_ATTR.hashCode();
    /**
     * Hash code for head label position attribute (head_lp).
     */
    public final static int HEADLP_HASH          = HEADLP_ATTR.hashCode();
    /**
     * Hash code for tail label attribute (taillabel).
     */
    public final static int TAILLABEL_HASH       = TAILLABEL_ATTR.hashCode();
    /**
     * Hash code for tail label position attribute (tail_lp).
     */
    public final static int TAILLP_HASH          = TAILLP_ATTR.hashCode();
    /**
     * Hash code for margin attribute (margin).
     */
    public final static int MARGIN_HASH      = MARGIN_ATTR.hashCode();
    /**
     * Hash code for mincross limit attribute (mclimit).
     */
    public final static int MCLIMIT_HASH     = MCLIMIT_ATTR.hashCode();
    /**
     * Hash code for minimum subgraph bounding box attribute (minbox).
     */
    public final static int MINBOX_HASH      = MINBOX_ATTR.hashCode();
    /**
     * Hash code for minimum rank distance between head and tail of edges attribute (minlen).
     */
    public final static int MINLEN_HASH      = MINLEN_ATTR.hashCode();
    /**
     * Hash code for minimum subgraph size attribute (minsize).
     */
    public final static int MINSIZE_HASH      = MINSIZE_ATTR.hashCode();
    /**
     * Hash code for node separation attribute (nodesep).
     */
    public final static int NODESEP_HASH     = NODESEP_ATTR.hashCode();
    /**
     * Hash code for orientation attribute (orientation).
     */
    public final static int ORIENTATION_HASH = ORIENTATION_ATTR.hashCode();
    /**
     * Hash code for patch work attribute (patch).
     */
    public final static int PATCH_HASH = PATCH_ATTR.hashCode();
    /**
     * Hash code for peripheries attribute (peripheries).
     */
    public final static int PERIPHERIES_HASH = PERIPHERIES_ATTR.hashCode();
    /**
     * Hash code for position attribute (pos).
     */
    public final static int POS_HASH         = POS_ATTR.hashCode();
    /**
     * Hash code for rank direction attribute (rankdir).
     */
    /**
     * Hash code for print list attribute (printlist).
     */
    public final static int PRINTLIST_HASH      = PRINTLIST_ATTR.hashCode();
    public final static int RANKDIR_HASH     = RANKDIR_ATTR.hashCode();
    /**
     * Hash code for rank separation attribute (ranksep).
     */
    public final static int RANKSEP_HASH     = RANKSEP_ATTR.hashCode();
    /**
     * Hash code for rectangles attribute (rects).
     */
    public final static int RECTS_HASH       = RECTS_ATTR.hashCode();
    /**
     * Hash code for rotation attribute (rotation).
     */
    public final static int ROTATION_HASH       = ROTATION_ATTR.hashCode();
    /**
     * Hash code for shape attribute (shape).
     */
    public final static int SHAPE_HASH       = SHAPE_ATTR.hashCode();
    /**
     * Hash code for sides attribute (sides).
     */
    public final static int SIDES_HASH       = SIDES_ATTR.hashCode();
    /**
     * Hash code for size attribute (size).
     */
    public final static int SIZE_HASH        = SIZE_ATTR.hashCode();
    /**
     * Hash code for skew attribute (skew).
     */
    public final static int SKEW_HASH        = SKEW_ATTR.hashCode();
    /**
     * Hash code for style attribute (style).
     */
    public final static int STYLE_HASH       = STYLE_ATTR.hashCode();
    /**
     *  Hash code for tag attribute (tag).
     */
    public final static int TAG_HASH         = TAG_ATTR.hashCode();
    /**
     *  Hash code for tip attribute (tip).
     */
    public final static int TIP_HASH         = TIP_ATTR.hashCode();
    /**
     * Hash code for weight attribute (weight).
     */
    public final static int WEIGHT_HASH      = WEIGHT_ATTR.hashCode();
    /**
     * Hash code for width attribute (width).
     */
    public final static int WIDTH_HASH       = WIDTH_ATTR.hashCode();

    //
    // Attribute types
    //

    /**
     * Indicator that no attribute value type is specified.
     * When no attribute type is specified, an error results.
     */
    public final static int  _NO_TYPE			= 0x00;
    /**
     * Indicator that attribute value is an instance of GrappaBox.
     */
    public final static int  BOX_TYPE			= 0x01;
    /**
     * Indicator that attribute value is an instance of java.awt.Color.
     */
    public final static int  COLOR_TYPE		= 0x02;
    /**
     * Indicator that attribute value is an instance of java.lang.Integer representing an edge direction.
     */
    public final static int DIR_TYPE		= 0x03;
    /**
     * Indicator that attribute value is an instance of java.lang.Double.
     */
    public final static int DOUBLE_TYPE		= 0x04;
    /**
     * Indicator that attribute value is a java.lang.Integer representing a font style.
     */
    public final static int FONTSTYLE_TYPE		= 0x05;
    /**
     * Indicator that attribute value is a java.lang.Hashtable whose keys provide a list of values
     */
    public final static int HASHLIST_TYPE		= 0x06;
    /**
     * Indicator that attribute value is an instance of java.lang.Integer.
     */
    public final static int INTEGER_TYPE		= 0x07;
    /**
     * Indicator that attribute value is an instance of GrappaLine.
     */
    public final static int  LINE_TYPE		= 0x08;
    /**
     * Indicator that attribute value is an instance of GrappaPoint.
     */
    public final static int  POINT_TYPE		= 0x09;
    /**
     * Indicator that attribute value is a java.lang.Integer representing a Grappa shape.
     */
    public final static int SHAPE_TYPE		= 0x0A;
    /**
     * Indicator that attribute value is an instance of GrappaSize.
     */
    public final static int SIZE_TYPE			= 0x0B;
    /**
     * Indicator that attribute value is an instance of java.lang.String.
     */
    public final static int STRING_TYPE		= 0x0C;
    /**
     * Indicator that attribute value is an instance of GrappaStyle.
     */
    public final static int STYLE_TYPE		= 0x0D;

    /*
     * The indicators used to define the underlying Shape of this object.
     */


    /**
     * Indicator that a valid shape was not specified for a graph element.
     */
    public final static int NO_SHAPE			= 0;
    /**
     * Indicator that the element has a line shape.
     */
    public final static int LINE_SHAPE			= 1;
    /**
     * Indicator that the element has a box shape.
     */
    public final static int BOX_SHAPE			= 2;
    /**
     * Indicator that the element has a diamond shape.
     */
    public final static int DIAMOND_SHAPE		= 3;
    /**
     * Indicator that the element has a double circle shape.
     */
    public final static int DOUBLECIRCLE_SHAPE		= 4;
    /**
     * Indicator that the element has a double octagon shape.
     */
    public final static int DOUBLEOCTAGON_SHAPE		= 5;
    /**
     * Indicator that the element has a egg shape.
     */
    public final static int EGG_SHAPE			= 6;
    /**
     * Indicator that the element has a hexagon shape.
     */
    public final static int HEXAGON_SHAPE		= 7;
    /**
     * Indicator that the element has a house shape.
     */
    public final static int HOUSE_SHAPE			= 8;
    /**
     * Indicator that the element has a upside-down house shape.
     */
    public final static int INVERTEDHOUSE_SHAPE		= 9;
    /**
     * Indicator that the element has a upside-down trapezium shape.
     */
    public final static int INVERTEDTRAPEZIUM_SHAPE	= 10;
    /**
     * Indicator that the element has a upside-down triangle shape.
     */
    public final static int INVERTEDTRIANGLE_SHAPE	= 11;
    /**
     * Indicator that the element has a octagon shape.
     */
    public final static int OCTAGON_SHAPE		= 12;
    /**
     * Indicator that the element has a oval shape.
     */
    public final static int OVAL_SHAPE			= 13;
    /**
     * Indicator that the element has a parallelogram shape.
     */
    public final static int PARALLELOGRAM_SHAPE		= 14;
    /**
     * Indicator that the element has a pentagon shape.
     */
    public final static int PENTAGON_SHAPE		= 15;
    /**
     * Indicator that the element has no shape, but rather is just a text label.
     */
    public final static int PLAINTEXT_SHAPE		= 16;
    /**
     * Indicator that the element has a general polygonal shape.
     */
    public final static int POINT_SHAPE		        = 17;
    /**
     * Indicator that the element has a general polygonal shape.
     */
    public final static int POLYGON_SHAPE		= 18;
    /**
     * Indicator that the element has a record shape.
     * A record shape is of a box shape that contains one or more labelled
     * sub-partitions within it.
     */
    public final static int RECORD_SHAPE		= 19;
    /**
     * Indicator that the element has a box shape with rounded corners.
     */
    public final static int ROUNDEDBOX_SHAPE		= 20;
    /**
     * Indicator that the element has a trapezium shape.
     */
    public final static int TRAPEZIUM_SHAPE		= 21;
    /**
     * Indicator that the element has a triangle shape.
     */
    public final static int TRIANGLE_SHAPE		= 22;
    /**
     * Indicator that the element has a triple octagon shape.
     */
    public final static int TRIPLEOCTAGON_SHAPE		= 23;
    /**
     * Indicator that the element has a circular shape with parallel chords top and bottom.
     */
    public final static int MCIRCLE_SHAPE		= 24;
    /**
     * Indicator that the element has a diamond shape with triangles inset in each corner.
     */
    public final static int MDIAMOND_SHAPE		= 25;
    /**
     * Indicator that the element has a record shape with triangles inset in each of its four outer corners.
     */
    public final static int MRECORD_SHAPE		= 26;
    /**
     * Indicator that the element has a square shape with triangles inset in each of its corners.
     */
    public final static int MSQUARE_SHAPE		= 27;
    /**
     * Indicator that the element shape is determined by a user-supplied class defined by the "custom" attribute
     */
    public final static int CUSTOM_SHAPE		= 28;

    // or'ed with others shape types
    /**
     * Bit mask for extracting shape information.
     */
    public final static int SHAPE_MASK			= 1023;
    /**
     * Bit flag indicating that the shape path needs to be generated by
     * by Grappa rather than relying on Java built-ins.
     */
    public final static int GRAPPA_SHAPE		= 1024;
}
