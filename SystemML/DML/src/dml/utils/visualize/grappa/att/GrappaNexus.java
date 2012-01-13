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

import java.lang.reflect.*;
import java.awt.*;
import java.awt.geom.*;
import java.awt.image.*;
import java.net.*;
import java.util.Observer;
import java.util.Hashtable;

/**
 * This class brings together shape, text and attribute information
 * related to bounding and drawing an element.
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public class GrappaNexus
    implements
	GrappaConstants,
	Cloneable, ImageObserver, Observer, Shape
{
    /**
     * RoundRectangle arc height factor
     */
    public static double arcHeightFactor = 0.05;

    /**
     * RoundRectangle arc width factor
     */
    public static double arcWidthFactor = 0.05;


    static {
        if (Grappa.toolkit == null) {
            try {
                Grappa.toolkit = java.awt.Toolkit.getDefaultToolkit();
            }
            catch(Throwable err) {
            }
        }
    }



    Area textArea = null;
    Shape shape = null;
    int shapeType = NO_SHAPE;
    Rectangle2D bbox = null;
    GrappaStyle style = null;
    Color fillcolor = null;
    Color color = null;
    Image image = null;
    boolean imageLoading = false;

    boolean dirty = false; // just for cluster subgraphs, now

    Stroke stroke = null;

    // used for RECORD_SHAPE/MRECORD_SHAPE only, so far
    private Object[] objs = null;

    // used when SHAPE_ATTR is CUSTOM_SHAPE
    private Object custom_shape = null;

    /**
     * Indicates if element text should be included in the element
     * bounding box. By default, it is.
     *
     * @see Grappa#shapeBoundText
     */
    public boolean boundText = true;
    /**
     * Indicates if the area bounding the element text should be
     * filled/outlined along with the element when being drawn.
     * By default, it is not.
     *
     * @see Grappa#shapeClearText
     */
    public boolean clearText = false;
    /**
     * Indicates if element text should be drawn when drawing the
     * element. By default, it is.
     *
     * @see Grappa#shapeDrawText
     */
    public boolean drawText  = true;

    Element element = null;

    long lastUpdate = 0;
    private long lastShapeUpdate = 0;
    private long lastTextUpdate = 0;
    private long lastStyleUpdate = 0;
    private long lastDecorationUpdate = 0;
    private long lastImageUpdate = 0;

    Font font = null;
    String[] lstr = null;
    GrappaPoint[] lpos = null;
    Color font_color = null;

    // fix winding rule at instantiation time
    private int windingRule = Grappa.windingRule;

    ////////////////////////////////////////////////////////////////////////
    //
    // Constructors
    //
    ////////////////////////////////////////////////////////////////////////

    /**
     * Constructs a new <code>GrappaNexus</code> object from an element.
     * @param elem the <code>Element</code> needing a <code>GrappaNexus</code> object.
     */
    public GrappaNexus(Element elem) {
	this.element = elem;
	rebuild();
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // Public methods
    //
    ////////////////////////////////////////////////////////////////////////

    /**
     * Get the underlying element.
     *
     * @return the element underlying this GrappaNexus.
     */
    public Element getElement() {
	return element;
    }

    /**
     * Return the image, if any, loaded for this element
     * @return an image or null
     */
    public Image getImage() {
	return image;
    }

    /**
     * Return status of image loading.
     * Returns true whenever an image has begun loading for this
     * element, but has not yet completed.
     *
     * @return true, during image loading; false, otherwise
     */
    public boolean isImageLoading() {
	return imageLoading;
    }

    /**
     * Return the winding rule for this line.
     * @return one of WIND_NON_ZERO or WIND_EVEN_ODD
     */
    public int getWindingRule() {
	return windingRule;
    }

    /**
     * Recompute the components of this GrappaNexus.
     *
     * @see updateStyle
     * @see updateDecoration
     * @see updateShape
     * @see updateText
     * @see updateImage
     */
    public void rebuild() {
	updateStyle();
	updateDecoration();
	updateShape();
	updateText();
	updateImage();
    }

    /**
     * Update the shape information for the underlying element.
     * For nodes, the <I>distortion</I>, <I>height</I>, <I>orientation</I>, <I>peripheries</I>, <I>pos</I>, <I>rotation</I>, <I>shape</I>, <I>sides</I>, <I>skew</I> and <I>width</I> attributes are examined.
     * For edges, the <I>pos</I> attribute is examined.
     * For subgraph, the bounding box is recomputed.
     */
    public void updateShape() {
	long thisShapeUpdate = System.currentTimeMillis();
	switch(element.getType()) {
	case NODE:
	    // re-initialize some values
	    custom_shape = null;
	    objs = null;

	    if(element.getSubgraph().isCluster() && element.getSubgraph().grappaNexus != null)
		element.getSubgraph().grappaNexus.dirty = true;

	    Node node = (Node)element;
	    GrappaPoint pos = (GrappaPoint)node.getAttributeValue(POS_ATTR);
	    Double Width = (Double)node.getAttributeValue(WIDTH_ATTR);
	    Double Height = (Double)node.getAttributeValue(HEIGHT_ATTR);
	    Integer Type = (Integer)node.getAttributeValue(SHAPE_ATTR);

	    double width = PointsPerInch * Width.doubleValue();
	    double height = PointsPerInch * Height.doubleValue();
	    int type = Type.intValue();

	    // the above attributes are sure to be there since they are defaulted,
	    // but these could return null values, so be sure to account for that
	    Integer Peripheries = (Integer)node.getAttributeValue(PERIPHERIES_ATTR);
	    Integer Sides = (Integer)node.getAttributeValue(SIDES_ATTR);
	    Double Distortion = (Double)node.getAttributeValue(DISTORTION_ATTR);
	    Double Skew = (Double)node.getAttributeValue(SKEW_ATTR);
	    Double Orientation = (Double)node.getAttributeValue(ORIENTATION_ATTR);
	    Double Rotation = (Double)node.getAttributeValue(ROTATION_ATTR);

	    int peripheries = Peripheries == null ? -1 : Peripheries.intValue();
	    int sides = Sides == null ? -1 : Sides.intValue();
	    double distortion = Distortion == null ? 0 : Distortion.doubleValue();
	    double skew = Skew == null ? 0 : Skew.doubleValue();
	    double orientation = Orientation == null ? 0 : Orientation.doubleValue();
	    double rotation = Rotation == null ? 0 : Rotation.doubleValue();

	    if(Orientation != null && orientation != 0 && Grappa.orientationInDegrees) {
		orientation = Math.PI * orientation / 180.0;
	    }

	    GeneralPath path;

	    switch(type) {
	    case CUSTOM_SHAPE:
		String custom = (String)node.getAttributeValue(CUSTOM_ATTR);
		if(custom == null) {
		    throw new IllegalArgumentException("custom attibuted null for node (" + node.getName() + ") with custom shape");
		}
		Class custom_class;
		try {
		    custom_class = Class.forName(custom);
		}
		catch(Exception e) {
		    throw new IllegalArgumentException("custom class unavailable for custom shape '" + custom + "'");
		}
		if(!(GrappaShape.class.isAssignableFrom(custom_class))) {
		    throw new IllegalArgumentException("custom class '" + custom + "' does not extend the GrappaShape class");
		}
		Constructor ccustom;
		try {
		    ccustom= custom_class.getConstructor(new Class[] { Element.class, double.class, double.class, double.class, double.class });
		}
		catch(Exception e) {
		    throw new IllegalArgumentException("constructor for custom class shape '" + custom + "' not found");
		}
		try {
		    if(Grappa.centerPointNodes) {
			shape = (Shape)(custom_shape = ccustom.newInstance(new Object[] { node,  new Double(pos.x - (width/2.0)), new Double(pos.y - (height/2.0)), new Double(width), new Double(height) }));
		    } else {
			shape = (Shape)(custom_shape = ccustom.newInstance(new Object[] { node, new Double(pos.x), new Double(pos.y), new Double(width), new Double(height) }));
		    }
		}
		catch(Exception e) {
		    if(e instanceof InvocationTargetException) {
			Throwable t = ((InvocationTargetException)e).getTargetException();
			Grappa.displayException((Exception)t);
		    } else if(e instanceof UndeclaredThrowableException) {
			throw new IllegalArgumentException("cannot instantiate custom shape '" + custom + "' for node '" + node.getName() + "' because2: " + ((UndeclaredThrowableException)e).getUndeclaredThrowable().getMessage());
		    } else {
			throw new IllegalArgumentException("cannot instantiate custom shape '" + custom + "' for node '" + node.getName() + "' because3: " + e.getMessage());
		    }
		}
		shapeType = CUSTOM_SHAPE;
		break;
	    case BOX_SHAPE:
		if(
		   (Distortion == null || distortion == 0)
		   &&
		   (Skew == null || skew == 0)
		   &&
		   (Orientation == null || orientation == 0)
		   ) {
		    shapeType = BOX_SHAPE;
		    if(Grappa.centerPointNodes) {
			shape = new Rectangle2D.Double(pos.x - (width/2.0), pos.y - (height/2.0), width, height);
		    } else {
			shape = new Rectangle2D.Double(pos.x, pos.y, width, height);
		    }
		    if(Peripheries != null && peripheries > 1) {
			path = new GeneralPath(shape);
			for(int i = 1; i < peripheries; i++) {
			    if(Grappa.centerPointNodes) {
				path.append
				    (
				     new Rectangle2D.Double
					 (
					  (pos.x - width/2.0) + (double)(i * PERIPHERY_GAP),
					  (pos.y - height/2.0) + (double)(i * PERIPHERY_GAP),
					  width - (double)(2 * i * PERIPHERY_GAP),
					  height - (double)(2 * i * PERIPHERY_GAP)
					  ),
				     false
				     );
			    } else {
				path.append
				    (
				     new Rectangle2D.Double
					 (
					  pos.x + (double)(i * PERIPHERY_GAP),
					  pos.y + (double)(i * PERIPHERY_GAP),
					  width - (double)(2 * i * PERIPHERY_GAP),
					  height - (double)(2 * i * PERIPHERY_GAP)
					  ),
				     false
				     );
			    }
			}
			shape = path;
		    }
		} else {
		    shapeType = BOX_SHAPE|GRAPPA_SHAPE;
		    if(Grappa.centerPointNodes) {
			shape = new GrappaShape(shapeType, pos.x, pos.y, width, height, sides, peripheries, distortion, skew, orientation, style.rounded, style.diagonals, null);
		    } else {
			shape = new GrappaShape(shapeType, pos.x + (width/2.0), pos.y + (height/2.0), width, height, sides, peripheries, distortion, skew, orientation, style.rounded, style.diagonals, null);
		    }
		}
		break;
	    case ROUNDEDBOX_SHAPE:
		if(
		   (Distortion == null || distortion == 0)
		   &&
		   (Skew == null || skew == 0)
		   &&
		   (Orientation == null || orientation == 0)
		   ) {
		    shapeType = ROUNDEDBOX_SHAPE;
		    if(Grappa.centerPointNodes) {
			shape = new RoundRectangle2D.Double(pos.x - (width/2.0), pos.y - (height/2.0), width, height, arcWidthFactor * width, arcHeightFactor * height);
		    } else {
			shape = new RoundRectangle2D.Double(pos.x, pos.y, width, height, arcWidthFactor * width, arcHeightFactor * height);
		    }
		    if(Peripheries != null && peripheries > 1) {
			path = new GeneralPath(shape);
			for(int i = 1; i < peripheries; i++) {
			    if(Grappa.centerPointNodes) {
				path.append
				    (
				     new RoundRectangle2D.Double
					 (
					  (pos.x - width/2.0) + (double)(i * PERIPHERY_GAP),
					  (pos.y - height/2.0) + (double)(i * PERIPHERY_GAP),
					  width - (double)(2 * i * PERIPHERY_GAP),
					  height - (double)(2 * i * PERIPHERY_GAP),
					  arcWidthFactor * (width - (double)(2 * i * PERIPHERY_GAP)),
					  arcHeightFactor * (height - (double)(2 * i * PERIPHERY_GAP))
					  ),
				     false
				     );
			    } else {
				path.append
				    (
				     new RoundRectangle2D.Double
					 (
					  pos.x + (double)(i * PERIPHERY_GAP),
					  pos.y + (double)(i * PERIPHERY_GAP),
					  width - (double)(2 * i * PERIPHERY_GAP),
					  height - (double)(2 * i * PERIPHERY_GAP),
					  arcWidthFactor * (width - (double)(2 * i * PERIPHERY_GAP)),
					  arcHeightFactor * (height - (double)(2 * i * PERIPHERY_GAP))
					  ),
				     false
				     );
			    }
			}
			shape = path;
		    }
		} else {
		    shapeType = ROUNDEDBOX_SHAPE|GRAPPA_SHAPE;
		    if(Grappa.centerPointNodes) {
			shape = new GrappaShape(shapeType, pos.x, pos.y, width, height, sides, peripheries, distortion, skew, orientation, style.rounded, style.diagonals, null);
		    } else {
			shape = new GrappaShape(shapeType, pos.x + (width/2.0), pos.y + (height/2.0), width, height, sides, peripheries, distortion, skew, orientation, style.rounded, style.diagonals, null);
		    }
		}
		break;
	    case OVAL_SHAPE:
		if(
		   (Distortion == null || distortion == 0)
		   &&
		   (Skew == null || skew == 0)
		   &&
		   (Orientation == null || orientation == 0)
		   ) {
		    shapeType = OVAL_SHAPE;
		    if(Grappa.centerPointNodes) {
			shape = new Ellipse2D.Double(pos.x - (width/2.0), pos.y - (height/2.0), width, height);
		    } else {
			shape = new Ellipse2D.Double(pos.x, pos.y, width, height);
		    }
		    if(Peripheries != null && peripheries > 1) {
			path = new GeneralPath(shape);
			for(int i = 1; i < peripheries; i++) {
			    if(Grappa.centerPointNodes) {
				path.append
				    (
				     new Ellipse2D.Double
					 (
					  (pos.x - width/2.0) + (double)(i * PERIPHERY_GAP),
					  (pos.y - height/2.0) + (double)(i * PERIPHERY_GAP),
					  width - (double)(2 * i * PERIPHERY_GAP),
					  height - (double)(2 * i * PERIPHERY_GAP)
					  ),
				     false
				     );
			    } else {
				path.append
				    (
				     new Ellipse2D.Double
					 (
					  pos.x + (double)(i * PERIPHERY_GAP),
					  pos.y + (double)(i * PERIPHERY_GAP),
					  width - (double)(2 * i * PERIPHERY_GAP),
					  height - (double)(2 * i * PERIPHERY_GAP)
					  ),
				     false
				     );
			    }
			}
			shape = path;
		    }
		} else {
		    shapeType = OVAL_SHAPE|GRAPPA_SHAPE;
		    if(Grappa.centerPointNodes) {
			shape = new GrappaShape(shapeType, pos.x, pos.y, width, height, sides, peripheries, distortion, skew, orientation, style.rounded, style.diagonals, null);
		    } else {
			shape = new GrappaShape(shapeType, pos.x + (width/2.0), pos.y + (height/2.0), width, height, sides, peripheries, distortion, skew, orientation, style.rounded, style.diagonals, null);
		    }
		}
		break;
	    case DIAMOND_SHAPE:
	    case DOUBLECIRCLE_SHAPE:
	    case DOUBLEOCTAGON_SHAPE:
	    case EGG_SHAPE:
	    case HEXAGON_SHAPE:
	    case HOUSE_SHAPE:
	    case INVERTEDHOUSE_SHAPE:
	    case INVERTEDTRAPEZIUM_SHAPE:
	    case INVERTEDTRIANGLE_SHAPE:
	    case OCTAGON_SHAPE:
	    case PARALLELOGRAM_SHAPE:
	    case PENTAGON_SHAPE:
	    case PLAINTEXT_SHAPE:
	    case POINT_SHAPE:
	    case POLYGON_SHAPE:
	    case TRAPEZIUM_SHAPE:
	    case TRIANGLE_SHAPE:
	    case TRIPLEOCTAGON_SHAPE:
	    case MCIRCLE_SHAPE:
	    case MDIAMOND_SHAPE:
	    case MSQUARE_SHAPE:
		shapeType = type|GRAPPA_SHAPE;
		if(Grappa.centerPointNodes) {
		    shape = new GrappaShape(shapeType, pos.x, pos.y, width, height, sides, peripheries, distortion, skew, orientation, style.rounded, style.diagonals, null);
		} else {
		    shape = new GrappaShape(shapeType, pos.x + (width/2.0), pos.y + (height/2.0), width, height, sides, peripheries, distortion, skew, orientation, style.rounded, style.diagonals, null);
		}
		break;
	    case RECORD_SHAPE:
	    case MRECORD_SHAPE:
		shapeType = type|GRAPPA_SHAPE;
		objs = GrappaSupportRects.parseRecordInfo(node);
		String rects = null;
		if(objs != null)
		    rects = (String)(objs[2]);
		if(Grappa.centerPointNodes) {
		    shape = new GrappaShape(shapeType, pos.x, pos.y, width, height, sides, peripheries, distortion, skew, orientation, style.rounded, style.diagonals, rects);
		} else {
		    shape = new GrappaShape(shapeType, pos.x + (width/2.0), pos.y + (height/2.0), width, height, sides, peripheries, distortion, skew, orientation, style.rounded, style.diagonals, rects);
		}
		break;
	    default:
		throw new IllegalArgumentException("unsupported type for this constructor (" + type + ")");
	    }

	    // handle rotation (rotation just spins the node,
	    // orientation spins it within a fixed bounding box
	    if(Rotation != null && rotation != 0 && shape != null) {
		double theta = rotation;
		if(Grappa.rotationInDegrees) {
		    theta = Math.PI * theta / 180.0;
		}
		if(Grappa.centerPointNodes)
		    shape = AffineTransform.getRotateInstance(theta,pos.x,pos.y).createTransformedShape(shape);
		else
		    shape = AffineTransform.getRotateInstance(theta,pos.x+(width/2.0),pos.y+(height/2.0)).createTransformedShape(shape);
	    }
	    break;
	case EDGE:
	    Edge edge = (Edge)element;
	    shapeType = LINE_SHAPE;

	    if(element.getSubgraph().isCluster() && element.getSubgraph().grappaNexus != null)
		element.getSubgraph().grappaNexus.dirty = true;

	    if((shape = (Shape)edge.getAttributeValue(POS_ATTR)) == null) {
		Integer attr_type = (Integer)(edge.getAttributeValue(DIR_ATTR));

		edge.direction = (attr_type != null ? attr_type.intValue() : (edge.getGraph().isDirected()?GrappaLine.TAIL_ARROW_EDGE:GrappaLine.NONE_ARROW_EDGE));

		// create a default straight line connecting the two
		// node centers
		edge.setAttribute(
				  "pos",
				  new GrappaLine(new GrappaPoint[] { (GrappaPoint)(edge.getTail().getAttributeValue(POS_ATTR)), (GrappaPoint)(edge.getHead().getAttributeValue(POS_ATTR)) }, edge.direction)
				      );
		shape = (Shape)edge.getAttributeValue(POS_ATTR);
	    }
	    break;
	case SUBGRAPH:
	    Subgraph subgraph = (Subgraph)element;
	    shapeType = BOX_SHAPE;

	    dirty = false;

	    // cannot call subgraph.getBoundingBox() because it would recurse,
	    // so just put the guts here
	    Rectangle2D sgbox = null;
	    Element elem = null;
	    GraphEnumeration enm = subgraph.elements();
	    while(enm.hasMoreElements()) {
		elem = enm.nextGraphElement();
		if(elem == element) continue;
		switch(elem.getType()) {
		case Grappa.NODE:
		case Grappa.EDGE:
		    elem.buildShape();
		    if(sgbox == null) {
			sgbox = elem.grappaNexus.getBounds2D();
		    } else {
			sgbox.add(elem.grappaNexus.rawBounds2D());
		    }
		    break;
		case Grappa.SUBGRAPH:
		    if(sgbox == null) {
			sgbox = ((Subgraph)elem).getBoundingBox();
		    } else {
			sgbox.add(((Subgraph)elem).getBoundingBox());
		    }
		    break;
		default: // cannot happen
		    throw new InternalError("unknown type (" + elem.getType() + ")");
		}
	    }
	    GrappaSize minSize = (GrappaSize)element.getAttributeValue(MINSIZE_ATTR);
	    if(minSize != null) {
		if(sgbox == null) {
		    sgbox = new java.awt.geom.Rectangle2D.Double(0,0,minSize.getWidth(),minSize.getHeight());
		} else {
		    sgbox.add(new java.awt.geom.Rectangle2D.Double(sgbox.getCenterX()-(minSize.getWidth()/2.0),sgbox.getCenterY()-(minSize.getHeight()/2.0),minSize.getWidth(),minSize.getHeight()));
		}
	    }
	    GrappaBox minBox = (GrappaBox)element.getAttributeValue(MINBOX_ATTR);
	    if(minBox != null) {
		if(sgbox == null) {
		    sgbox = new java.awt.geom.Rectangle2D.Double(minBox.x,minBox.y,minBox.width,minBox.height);
		} else {
		    sgbox.add(new java.awt.geom.Rectangle2D.Double(minBox.x,minBox.y,minBox.width,minBox.height));
		}
	    }
	    if(sgbox == null) {
		sgbox = new java.awt.geom.Rectangle2D.Double(0,0,0,0);
	    }
	    shape = sgbox;
	    break;
	default:
	    throw new IllegalArgumentException("unrecognized element type (" + element.getType() + ") for " + element.getName());
	}

	bboxCheckSet();
	lastUpdate = lastShapeUpdate = thisShapeUpdate;
    }

    /**
     * Update the shape information for the underlying element.
     * The <I>style</I> attribute is examined.
     */
    public void updateStyle() {
	long thisStyleUpdate = System.currentTimeMillis();
	if((style = (GrappaStyle)element.getAttributeValue(STYLE_ATTR)) == null) {
	    throw new InternalError("style defaults not properly set in Graph.java");
	}

	// an attempt to handle font info passed via style instead of fontstyle
	if(
	   style.font_style != null
	   &&
	   style.font_style != (Integer)element.getAttributeValue(FONTSTYLE_ATTR)
	   ) {
	    element.setAttribute(FONTSTYLE_ATTR,style.font_style);
	    style.font_style = null;
	}
	lastUpdate = lastStyleUpdate = thisStyleUpdate;
    }

    /**
     * Update the text information for the underlying element.
     * The <I>fontcolor</I>, <I>fontname</I>, <I>fontsize</I>, <I>fontstyle</I>, and <I>label</I> attributes are examined.
     * The <I>lp</I> attribute is also examined for edges and subgraphs.
     */
    public void updateText() {
	String[] tstr = null;
	GrappaPoint[] tpos = null;
	Area area = null;
	Font tfont = null;
	boolean makeAdjustment = false;
	double signum = -1;
	int offset = 0;
	String headstr, tailstr;
	GrappaPoint headpt, tailpt;
	int lcnt = 0;
	boolean hasEdgeLabel = false;
	Attribute attr = null;

	long thisTextUpdate = System.currentTimeMillis();

	String[] labels;
	GrappaPoint[] lps;

	String labelAttr = (String)element.getAttributeValue(LABEL_ATTR);

	if(labelAttr != null && labelAttr.equals("\\N"))
	    labelAttr = element.getName();

	if(labelAttr != null && labelAttr.length() > 0) {
	    lcnt++;
	} else labelAttr = null;

	headstr = tailstr = null;
	headpt = tailpt = null;

	if (element.isEdge()) {
	    if ((headstr = (String)element.getAttributeValue(HEADLABEL_ATTR)) != null && (attr = element.getLocalAttribute(HEADLP_ATTR)) != null) {
		headpt = (GrappaPoint)(attr.getValue());
		lcnt++;
		hasEdgeLabel = true;
	    } else headstr = null;
	    if ((tailstr = (String)element.getAttributeValue(TAILLABEL_ATTR)) != null && (attr = element.getLocalAttribute(TAILLP_ATTR)) != null) {
		tailpt = (GrappaPoint)(attr.getValue());
		lcnt++;
		hasEdgeLabel = true;
	    } else tailstr = null;
	}

	// all string attributes are trimmed when they are stored
	// if(labelAttr != null)
	//     labelAttr = labelAttr.trim();

	if(labelAttr != null || hasEdgeLabel) {
	    if(
	       element.isNode()
	       &&
	       (
		shapeType == (RECORD_SHAPE|GRAPPA_SHAPE)
		||
		shapeType == (MRECORD_SHAPE|GRAPPA_SHAPE)
		)
	       &&
	       labelAttr.indexOf('|') < 0
	       &&
	       labelAttr.indexOf('{') == 0
	       &&
	       labelAttr.lastIndexOf('}') == labelAttr.length() - 1
	       ) {
		labelAttr = labelAttr.substring(1,labelAttr.length()-1).trim();
	    }

	    if(hasEdgeLabel || labelAttr.length() > 0) {

		String fontname = (String)element.getAttributeValue(FONTNAME_ATTR);
		Integer fontstyle = (Integer)element.getAttributeValue(FONTSTYLE_ATTR);
		Integer fontsize = (Integer)element.getAttributeValue(FONTSIZE_ATTR);
		Integer fontadj = (Integer)(element.getGraph()).getGrappaAttributeValue(GRAPPA_FONTSIZE_ADJUSTMENT_ATTR);

		// set font
		tfont = new Font(fontname,fontstyle.intValue(),fontsize.intValue() + fontadj.intValue());

		String rectString = null;

		int lines;
		int i;
		char[] array;
		int[] justification;
		Rectangle2D[] bnds;
		java.awt.font.LineMetrics[] mtrc;
		int start;
		char ch;
		String str;
		double wdinfo, htinfo;
		double top;
		double x;

		if(
		   element.isNode()
		   &&
		   (
		    shapeType == (RECORD_SHAPE|GRAPPA_SHAPE)
		    ||
		    shapeType == (MRECORD_SHAPE|GRAPPA_SHAPE)
		    )
		   &&
		   labelAttr.indexOf('|') >= 0
		   ) {
		    if(objs == null)
			updateShape();
		    if(objs != null && objs[0] != null && objs[1] != null) {
			labels = (String[])objs[0];
			lps = (GrappaPoint[])objs[1];
		    } else {
			labels = new String[1];
			labels[0] = labelAttr;
			lps = new GrappaPoint[1];
			lps[0] = ((Node)element).getCenterPoint();
		    }
		} else {
		    Subgraph sg;

		    labels = new String[lcnt];
		    lps = new GrappaPoint[lcnt];

		    lcnt = 0;
		    if (labelAttr != null)
			labels[lcnt++] = labelAttr;
		    if (headstr != null) {
			labels[lcnt] = headstr;
			lps[lcnt] = headpt;
			lcnt++;
		    }
		    if (tailstr != null) {
			labels[lcnt] = tailstr;
			lps[lcnt] = tailpt;
		    }

		    if (labelAttr != null) {
			if(Grappa.autoPositionNodeLabel && element.isNode()) {
			    lps[0] = ((Node)element).getCenterPoint();
			} else if((attr = element.getLocalAttribute(LP_ATTR)) == null || ((sg = element.getSubgraph()) != null && attr == sg.getLocalAttribute(LP_ATTR))) {
			    Rectangle2D lbox;

			    if((lbox = (Rectangle2D)element.getAttributeValue(BBOX_ATTR)) == null) {
				lbox = bbox;
			    }

			    if(element.isSubgraph() && lbox != null) {
				lps[0] = new GrappaPoint(lbox.getX() + lbox.getWidth()/2.0, (Grappa.labelGraphBottom?lbox.getMaxY():lbox.getMinY()));
				element.setAttribute(LP_ATTR, lps[0].clone());
			    } else {
				lps[0] = null;
			    }
			} else {
			    lps[0] = (GrappaPoint)(attr.getValue());
			}
			if(element.isSubgraph() && attr == null) {
			    if(Grappa.labelGraphBottom) {
				signum = 1;
			    } else {
				signum = -1;
			    }
			    makeAdjustment = true;
			}
		    }
		}

		for(int l = 0; l < labels.length; l++) {
		    if(labels[l] != null && lps[l] != null && labels[l].length() > 0) {

			if(labels[l].equals("\\N")) {
			    labels[l] = element.getName();
			    if(labels[l] == null) continue;
			}

			// break label into multiple lines as indicated by line-breaks
			lines = 1;
			array = labels[l].toCharArray();

			// first count lines
			for(i=0; i<array.length; i++) {
			    if(
			       array[i] == '\\'
			       &&
			       ++i < array.length
			       &&
			       (array[i] == 'l' || array[i] == 'r' || array[i] == 'n')
			       &&
			       (i+1) < array.length
			       ) {
				lines++;
			    }
			}

			if(tpos == null) {
			    offset = 0;
			    tpos = new GrappaPoint[lines];
			    tstr = new String[lines];
			} else {
			    offset = tpos.length;
			    GrappaPoint[] p = new GrappaPoint[offset+lines];
			    String[] s = new String[offset+lines];
			    System.arraycopy(tpos,0,p,0,offset);
			    System.arraycopy(tstr,0,s,0,offset);
			    tpos = p;
			    tstr = s;
			}
			justification = new int[lines];
			bnds = new Rectangle2D[lines];
			mtrc = new java.awt.font.LineMetrics[lines];

			// now extract lines and justification info
			lines = 0;
			start = 0;
			ch = 'n';
			str = null;
			wdinfo = htinfo = 0;
			for(i=0; i<array.length; i++) {
			    if(
			       array[i] == '\\'
			       &&
			       ++i < array.length
			       &&
			       ((ch = array[i]) == 'l' || array[i] == 'r' || array[i] == 'n')
			       ) {
				tstr[offset+lines] = new String(array,start,i-1-start);
				bnds[lines] = tfont.getStringBounds(tstr[offset+lines],element.getGraph().REFCNTXT);
				mtrc[lines] = tfont.getLineMetrics(tstr[offset+lines],element.getGraph().REFCNTXT);
				if(bnds[lines].getWidth() > wdinfo) wdinfo = bnds[lines].getWidth();
				//htinfo += bnds[lines].getHeight();
				htinfo += 2 + tfont.getSize();
				if(ch == 'l') justification[lines++] = -1;
				else if(ch == 'r') justification[lines++] = 1;
				else justification[lines++] = 0;
				start = (i+1);
			    }
			}
			if(start < array.length) {
			    tstr[offset+lines] = new String(array,start,array.length - start);
			    bnds[lines] = tfont.getStringBounds(tstr[offset+lines],element.getGraph().REFCNTXT);
			    mtrc[lines] = tfont.getLineMetrics(tstr[offset+lines],element.getGraph().REFCNTXT);
			    if(bnds[lines].getWidth() > wdinfo) wdinfo = bnds[lines].getWidth();
			    //htinfo += bnds[lines].getHeight();
			    htinfo += 2 + tfont.getSize();
			    if(ch == 'l') justification[lines] = -1;
			    else if(ch == 'r') justification[lines] = 1;
			    else justification[lines] = 0;
			}

			//htinfo += mtrc[lines].getLeading();
			//htinfo += (mtrc[lines].getLeading()) * tfont.getSize2D() / mtrc[lines].getHeight();

			if(makeAdjustment) {
			    if(Grappa.labelGraphOutside) {
				lps[l].y += signum * htinfo;
			    } else {
				lps[l].y -= signum * htinfo;
			    }
			}

			// half these as that's how they will be used
			wdinfo /= 2.0;
			htinfo /= 2.0;

			// figure out textArea and positioning of each line of text
			// doing it now instead of at rendering time means some
			// approximation, but text rendering is iffy anyway and this
			// will be close enough and more efficient (if you call this
			// efficient)

			// first, find top of text bounding box
			top = lps[l].y - htinfo;

			// now, for each line:
			// 1. determine left-side position and create (add to) textArea
			// 2. getAscent() to determine actually draw position
			// 3. shift down hieght of line
			x = 0;
			for(i = 0; i < bnds.length; i++) {
			    if(justification[i] < 0) {
				// left
				x = lps[l].x - wdinfo;
			    } else if(justification[i] > 0) {
				// right
				x = lps[l].x + wdinfo - bnds[i].getWidth();
			    } else {
				x = lps[l].x - bnds[i].getWidth()/2.0;
			    }
			    bnds[i].setRect(x,top,bnds[i].getWidth(),bnds[i].getHeight());
			    if(area == null) {
				area = new Area(bnds[i]);
			    } else {
				area.add(new Area(bnds[i]));
			    }
			    //tpos[offset+i] = new GrappaPoint(x, top + mtrc[i].getAscent());
			    //tpos[offset+i] = new GrappaPoint(x, top + Math.ceil((mtrc[i].getAscent()+mtrc[i].getLeading())*tfont.getSize()/mtrc[i].getHeight()));
			    tpos[offset+i] = new GrappaPoint(x, top + tfont.getSize() - 1);

			    //top += bnds[i].getHeight();
			    top += 2 + tfont.getSize();
			}
		    }
		}
	    }
	}

	// commit changes
	font = tfont;
	lpos = tpos;
	lstr = tstr;
	textArea = area;
	bboxCheckSet();
	lastUpdate = lastTextUpdate = thisTextUpdate;
    }

    /**
     * Update the decoration information for the underlying element.
     * The <I>color</I> and <I>fontcolor</I> attributes are examined.
     * For edges, the <I>dir</I> attribute is examined.
     */
    public void updateDecoration() {
	long thisDecorationUpdate = System.currentTimeMillis();
	color = (Color)(element.getAttributeValue(COLOR_ATTR));
	fillcolor = (Color)(element.getAttributeValue(FILLCOLOR_ATTR));
	font_color = (Color)(element.getAttributeValue(FONTCOLOR_ATTR));
	if(element.isEdge() && shape != null && shape instanceof GrappaLine) {
	    Edge edge = (Edge)element;
	    int graph_dir = edge.getGraph().isDirected() ? GrappaLine.TAIL_ARROW_EDGE : GrappaLine.NONE_ARROW_EDGE;
	    int dir = graph_dir;
	    Integer attr_type = (Integer)(edge.getThisAttributeValue(DIR_ATTR));
	    if(attr_type != null)
		dir = attr_type.intValue(); 

	    edge.direction = dir;

	    GrappaLine gline = (GrappaLine)shape;
	    boolean forward = gline.startsNear((Point2D)(edge.getTail().getAttributeValue(POS_ATTR))); 
	    // basically, it edge loops on same node, assume it is always
	    // in the forward orientation
	    if(!forward && edge.getHead() == edge.getTail())
		forward = true;

	    int line_dir;
	    if(forward) {
		line_dir = gline.getArrowType();
	    } else {
		switch(gline.getArrowType()) {
		case GrappaLine.HEAD_ARROW_EDGE:
		    line_dir = GrappaLine.TAIL_ARROW_EDGE;
		case GrappaLine.TAIL_ARROW_EDGE:
		    line_dir = GrappaLine.HEAD_ARROW_EDGE;
		    break;
		default:
		    line_dir = gline.getArrowType();
		    break;
		}
	    }
	    if(line_dir != dir) {
		if(forward) {
		    line_dir = dir;
		} else {
		    switch(dir) {
		    case GrappaLine.HEAD_ARROW_EDGE:
			line_dir = GrappaLine.TAIL_ARROW_EDGE;
		    case GrappaLine.TAIL_ARROW_EDGE:
			line_dir = GrappaLine.HEAD_ARROW_EDGE;
			break;
		    default:
			line_dir = dir;
			break;
		    }
		}
		gline.changeArrowType(line_dir);
		edge.setAttribute(POS_ATTR, gline);
	    }
	}
	lastUpdate = lastDecorationUpdate = thisDecorationUpdate;
    }

    /**
     * Update the image information for the underlying element.
     */
    public void updateImage() {
	long thisImageUpdate = System.currentTimeMillis();
	String path = (String)(element.getAttributeValue(IMAGE_ATTR));

	if(path != null && Grappa.toolkit != null) {

	    this.image = null;
	    imageLoading = true;

	    Image raw_image = null;

	    try {
		URL url = new URL(path);
		raw_image = Grappa.toolkit.getImage(url);
	    }
	    catch(Exception ex) {}

	    if(raw_image == null) {
		try {
		    raw_image = Grappa.toolkit.getImage(path);
		}
		catch(Exception ex) {}
	    }

	    if(raw_image != null) {
		if(Grappa.toolkit.prepareImage(raw_image,-1,-1,this)) {
		    this.image = raw_image;
		    imageLoading = false;
		}
	    } else {
		imageLoading = false;
	    }
	} else {
	    this.image = null;
	    imageLoading = false;
	}

	lastUpdate = lastImageUpdate = thisImageUpdate;
    }

    public final boolean
    imageUpdate(Image image, int flags, int x, int y, int width, int height) {

	boolean ret = true;

	synchronized(this) {
	    if((flags&ALLBITS) == ALLBITS) {
		ret = false;
		this.image = image;
		imageLoading = false;
		notifyAll();
	    } else if((flags&(ABORT|ERROR)) != 0) {
		ret = false;
		imageLoading = false;
		notifyAll();
	    }
	}

	return(ret);
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // Private methods
    //
    ////////////////////////////////////////////////////////////////////////

    private void bboxCheckSet() {
	Rectangle2D oldbox = bbox;
	bbox = null;
	Rectangle2D newbox = null;
	try {
	    newbox = rawBounds2D();
	}
	catch(Exception ex) {
	    throw (RuntimeException)(ex.fillInStackTrace());
	}
	finally {
	    bbox = oldbox;
	}

	if(newbox == null) {
	    if(element.isSubgraph() && ((Subgraph)element).countOfElements(SUBGRAPH|NODE|EDGE) == 0) {
		newbox = new Rectangle2D.Double();
	    } else {
		throw new InternalError("new bounding box of \"" + element.getName() + "\" is null");
	    }
	}
	if(
	   (oldbox == null && newbox != null)
	   ||
	   (oldbox != null && newbox == null)
	   ||
	   (newbox != null && !newbox.equals(oldbox))
	   ) {
	    // bounding box has changed so null out existing bboxes of enclosing subgraphs
	    Subgraph prnt = element.getSubgraph();
	    while(prnt != null) {
		if(prnt.grappaNexus != null) {
		  prnt.grappaNexus.bbox = null;
		}
		prnt = prnt.getSubgraph();
	    }

	    // commit
	    bbox = newbox;
	    lastUpdate = System.currentTimeMillis();
	}
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // Cloneable interface
    //
    ////////////////////////////////////////////////////////////////////////
 
    /**
     * Creates a new object of the same class as this object.
     *
     * @return     a clone of this instance.
     * @exception  OutOfMemoryError            if there is not enough memory.
     * @see        java.lang.Cloneable
     */
    public Object clone() {
	try {
	    GrappaNexus copy = (GrappaNexus) super.clone();
	    if(shape != null) {
		if(shapeType == LINE_SHAPE) {
		    copy.shape = (Shape) ((GrappaLine)shape).clone();
		} else if(shapeType == BOX_SHAPE) {
		    copy.shape = (Shape) ((Rectangle2D)shape).clone();
		} else if(shapeType == ROUNDEDBOX_SHAPE) {
		    copy.shape = (Shape) ((RoundRectangle2D)shape).clone();
		} else if(shapeType == OVAL_SHAPE) {
		    copy.shape = (Shape) ((Ellipse2D)shape).clone();
		} else if((shapeType&GRAPPA_SHAPE) != 0) {
		    copy.shape = (Shape) ((GrappaShape)shape).clone();
		} else {
		    copy.shape = (Shape) ((GeneralPath)shape).clone();
		}
	    }
	    if(textArea != null) {
		copy.textArea = (Area) textArea.clone();
	    }
	    return copy;
	} catch (CloneNotSupportedException e) {
	    // this shouldn't happen, since we are Cloneable
	    throw new InternalError();
	}
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // Shape interface
    //
    ////////////////////////////////////////////////////////////////////////

    public boolean contains(double x, double y) {

	boolean contains = false;

	if(shape != null) {
	    contains = shape.contains(x, y);
	}

	if(
	   textArea != null && !contains && !clearText && drawText
	   &&
	   (
	    (element.isNode() && element.getGraph().getShowNodeLabels())
	    ||
	    (element.isEdge() && element.getGraph().getShowEdgeLabels())
	    ||
	    (element.isSubgraph() && element.getGraph().getShowSubgraphLabels())
	    )
	   ) {
	    contains = textArea.contains(x, y);
	}

	return(contains);
    }

    public boolean contains(double x, double y, double width, double height) {

	boolean contains = false;

	if(shape != null) {
	    contains = shape.contains(x, y, width, height);
	}

	if(
	   textArea != null && !contains && !clearText && drawText
	   &&
	   (
	    (element.isNode() && element.getGraph().getShowNodeLabels())
	    ||
	    (element.isEdge() && element.getGraph().getShowEdgeLabels())
	    ||
	    (element.isSubgraph() && element.getGraph().getShowSubgraphLabels())
	    )
	   ) {
	    contains = textArea.contains(x, y, width, height);
	}

	return(contains);
    }

    public boolean contains(Point2D p) {

	return(contains(p.getX(),p.getY()));
    }

    public boolean contains(Rectangle2D r) {

	return(contains(r.getX(), r.getY(), r.getWidth(), r.getHeight()));
    }

    public Rectangle getBounds() {

	return(getBounds2D().getBounds());
    }

    public Rectangle2D getBounds2D() {

	if(dirty) {
	    bbox = null;
	    updateShape();
	}

	if(bbox == null) {

	    if(shape != null) {
		bbox = shape.getBounds2D();
	    }

	    if(textArea != null && Grappa.shapeBoundText && boundText) {
		if(bbox == null) {
		    bbox = textArea.getBounds();
		} else {
		    bbox.add(textArea.getBounds());
		}
	    }

	    if(bbox.getHeight() == 0)
		bbox.setRect(bbox.getX(),bbox.getY(),bbox.getWidth(),0.01);
	    if(bbox.getWidth() == 0)
		bbox.setRect(bbox.getX(),bbox.getY(),0.01,bbox.getHeight());

	}
	// return a clone so no one can mess with the real values
	return((Rectangle2D)(bbox.clone()));
    }

    // variation for use in Grappa
    Rectangle2D rawBounds2D() {

	if(dirty) {
	    bbox = null;
	    updateShape();
	}

	if(bbox == null) {

	    if(shape != null) {
		bbox = shape.getBounds2D();
	    }

	    if(textArea != null && Grappa.shapeBoundText && boundText) {
		if(bbox == null) {
		    bbox = textArea.getBounds();
		} else {
		    bbox.add(textArea.getBounds());
		}
	    }

	    if(bbox.getHeight() == 0)
		bbox.setRect(bbox.getX(),bbox.getY(),bbox.getWidth(),0.01);
	    if(bbox.getWidth() == 0)
		bbox.setRect(bbox.getX(),bbox.getY(),0.01,bbox.getHeight());
	}
	return(bbox);
    }

    /**
     * Equivalent to <TT>getPathIterator(null)</TT>.
     *
     * @see getPathIterator(AffineTransform)
     */
    public PathIterator getPathIterator() {
	return new GrappaPathIterator(this, null);
    }

    public PathIterator getPathIterator(AffineTransform at) {
	return new GrappaPathIterator(this, at);
    }

    public PathIterator getPathIterator(AffineTransform at, double flatness) {
	return new FlatteningPathIterator(new GrappaPathIterator(this, at), flatness);
    }

    public boolean intersects(double x, double y, double width, double height) {

	boolean intersects = false;

	if(shape != null) {
	    intersects = shape.intersects(x, y, width, height);
	}

	if(
	   textArea != null && !intersects && !clearText && drawText
	   &&
	   (
	    (element.isNode() && element.getGraph().getShowNodeLabels())
	    ||
	    (element.isEdge() && element.getGraph().getShowEdgeLabels())
	    ||
	    (element.isSubgraph() && element.getGraph().getShowSubgraphLabels())
	    )
	   ) {
	    intersects = textArea.intersects(x, y, width, height);
	}

	return(intersects);
    }

    public boolean intersects(Rectangle2D r) {

	return(intersects(r.getX(), r.getY(), r.getWidth(), r.getHeight()));
    }
   
    ////////////////////////////////////////////////////////////////////////
    //
    // Observer interface
    //
    ////////////////////////////////////////////////////////////////////////

    /**
     * This method is called whenever the observed object is changed.
     * When certain observed attributes (attributes of interest)
     * are changed, this method will update the GrappaNexus as needed.
     *
     * @param obs the Observable must be an Attribute
     * @param arg either a Long giving the update time of the Attribute as returned by System.getTimeInMillis() or it is a two element Object array, where the first element is a new Attribute to be observed in place of that passed via <I>obs</I> and the second element is the update time of this new Attribute.
     */
    public void update(java.util.Observable obs, Object arg) {

	// begin boilerplate
	if(!(obs instanceof Attribute)) {
	    throw new IllegalArgumentException("expected to be observing attributes only (obs) for \"" + element.getName() + "\"");
	}
	Attribute attr = (Attribute)obs;
	if(arg instanceof Object[]) {
	    Object[] args = (Object[])arg;
	    if(args.length == 2 && args[0] instanceof Attribute && args[1] instanceof Long) {
		attr.deleteObserver(this);
		attr = (Attribute)args[0];
		attr.addObserver(this);
		// in case we call: super.update(obs,arg)
		obs = attr;
		arg = args[1];
	    } else {
		throw new IllegalArgumentException("badly formated update information for \"" + element.getName() + "\"");
	    }
	}
	// end boilerplate


	// when this object is created it should register with the
	// appropriate Attributes based on how it was created;
	// this method  will then see what has been updated
	// and set flags in this object (and put tokens in an update
	// stack in Graph is autoUpdate is true) so that the appropriate
	// parts will be updated before any drawing occurs.
	if(arg instanceof Long) {

	    String attrName = attr.getName();
	    int attrHash = attr.getNameHash();
	    long thisUpdate = ((Long)arg).longValue() + 1L;

	    if(element == null || !element.reserve()) return;

	    // reset
	    objs = null;

	    switch(element.getType()) {
	    case NODE:
		if(
		   (POS_HASH == attrHash && POS_ATTR.equals(attrName))
		   ||
		   (WIDTH_HASH == attrHash && WIDTH_ATTR.equals(attrName))
		   ||
		   (HEIGHT_HASH == attrHash && HEIGHT_ATTR.equals(attrName))
		   ||
		   (SHAPE_HASH == attrHash && SHAPE_ATTR.equals(attrName))
		   ) {
		    if(lastShapeUpdate < thisUpdate) {
			updateShape();
			if(Grappa.autoPositionNodeLabel) {
			    updateText();
			}
		    }
		} else if(
			  (LABEL_HASH == attrHash && LABEL_ATTR.equals(attrName))
			  ||
			  (
			   !Grappa.autoPositionNodeLabel
			   &&
			   (LP_HASH == attrHash && LP_ATTR.equals(attrName)) // in case it is used
			   )
			  ||
			  (FONTSIZE_HASH == attrHash && FONTSIZE_ATTR.equals(attrName))
			  ||
			  (FONTNAME_HASH == attrHash && FONTNAME_ATTR.equals(attrName))
			  ||
			  (FONTSTYLE_HASH == attrHash && FONTSTYLE_ATTR.equals(attrName))
			  ) {
		    if(lastTextUpdate < thisUpdate) {
			updateText();
		    }
		} else if(
			  (STYLE_HASH == attrHash && STYLE_ATTR.equals(attrName))
			  ) {
		    if(lastStyleUpdate < thisUpdate) {
			updateStyle();
		    }
		} else if(
			  (COLOR_HASH == attrHash && COLOR_ATTR.equals(attrName))
			  ||
			  (FONTCOLOR_HASH == attrHash && FONTCOLOR_ATTR.equals(attrName))
			  ) {
		    if(lastDecorationUpdate < thisUpdate) {
			updateDecoration();
		    }
		} else if(
			  (IMAGE_HASH == attrHash && IMAGE_ATTR.equals(attrName))
			  ) {
		    if(lastImageUpdate < thisUpdate) {
			updateImage();
		    }
		} else {
		    throw new InternalError("update called for \"" + element.getName() + "\" with an unmonitored attribute: " + attrName);
		}
		break;
	    case EDGE:
		if(
		   (POS_HASH == attrHash && POS_ATTR.equals(attrName))
		   ) {
		    if(lastShapeUpdate < thisUpdate) {
			updateShape();
		    }
		} else if(
			  (LABEL_HASH == attrHash && LABEL_ATTR.equals(attrName))
			  ||
			  (LP_HASH == attrHash && LP_ATTR.equals(attrName))
			  ||
			  (HEADLABEL_HASH == attrHash && HEADLABEL_ATTR.equals(attrName))
			  ||
			  (HEADLP_HASH == attrHash && HEADLP_ATTR.equals(attrName))
			  ||
			  (TAILLABEL_HASH == attrHash && TAILLABEL_ATTR.equals(attrName))
			  ||
			  (TAILLP_HASH == attrHash && TAILLP_ATTR.equals(attrName))
			  ||
			  (FONTSIZE_HASH == attrHash && FONTSIZE_ATTR.equals(attrName))
			  ||
			  (FONTNAME_HASH == attrHash && FONTNAME_ATTR.equals(attrName))
			  ||
			  (FONTSTYLE_HASH == attrHash && FONTSTYLE_ATTR.equals(attrName))
			  ) {
		    if(lastTextUpdate < thisUpdate) {
			updateText();
		    }
		} else if(
			  (STYLE_HASH == attrHash && STYLE_ATTR.equals(attrName))
			  ) {
		    if(lastStyleUpdate < thisUpdate) {
			updateStyle();
		    }
		} else if(
			  (COLOR_HASH == attrHash && COLOR_ATTR.equals(attrName))
			  ||
			  (DIR_HASH == attrHash && DIR_ATTR.equals(attrName))
			  ||
			  (FONTCOLOR_HASH == attrHash && FONTCOLOR_ATTR.equals(attrName))
			  ) {
		    if(lastDecorationUpdate < thisUpdate) {
			updateDecoration();
		    }
		} else if(
			  (IMAGE_HASH == attrHash && IMAGE_ATTR.equals(attrName))
			  ) {
		    if(lastImageUpdate < thisUpdate) {
			updateImage();
		    }
		} else {
		    throw new InternalError("update called for \"" + element.getName() + "\" with an unmonitored attribute: " + attrName);
		}
		break;
	    case SUBGRAPH:
		if(
		   (LABEL_HASH == attrHash && LABEL_ATTR.equals(attrName))
		   ||
		   (LP_HASH == attrHash && LP_ATTR.equals(attrName))
		   ||
		   (FONTSIZE_HASH == attrHash && FONTSIZE_ATTR.equals(attrName))
		   ||
		   (FONTNAME_HASH == attrHash && FONTNAME_ATTR.equals(attrName))
		   ||
		   (FONTSTYLE_HASH == attrHash && FONTSTYLE_ATTR.equals(attrName))
		   ) {
		    if(lastTextUpdate < thisUpdate) {
			updateText();
		    }
		} else if(
			  (STYLE_HASH == attrHash && STYLE_ATTR.equals(attrName))
			  ) {
		    if(lastStyleUpdate < thisUpdate) {
			updateStyle();
		    }
		} else if(
			  (COLOR_HASH == attrHash && COLOR_ATTR.equals(attrName))
			  ||
			  (FONTCOLOR_HASH == attrHash && FONTCOLOR_ATTR.equals(attrName))
			  ) {
		    if(lastDecorationUpdate < thisUpdate) {
			updateDecoration();
		    }
		} else if(
			  (IMAGE_HASH == attrHash && IMAGE_ATTR.equals(attrName))
			  ) {
		    if(lastImageUpdate < thisUpdate) {
			updateImage();
		    }
		} else if(
			  (MINBOX_HASH == attrHash && MINBOX_ATTR.equals(attrName))
			  ||
			  (MINSIZE_HASH == attrHash && MINSIZE_ATTR.equals(attrName))
			  ) {
		    bbox = null;
		} else {
		    throw new InternalError("update called for \"" + element.getName() + "\" with an unmonitored attribute: " + attrName);
		}
		break;
	    }

	    element.release();
	} else {
	    throw new InternalError("update called for shape of element \"" + element.getName() + "\" without proper format");
	}
    }

    /**
     * Draw the element using the supplied Graphics2D context.
     *
     * @param g2d the Graphics2D context to be used for drawing
     */
    void draw(java.awt.Graphics2D g2d) {
	if(shape instanceof CustomRenderer)
	    ((CustomRenderer)shape).draw(g2d);
	else
	    g2d.draw(this);
    }

    /**
     * Fill the element using the supplied Graphics2D context.
     *
     * @param g2d the Graphics2D context to be used for drawing
     */
    void fill(java.awt.Graphics2D g2d) {
	if(shape instanceof CustomRenderer)
	    ((CustomRenderer)shape).fill(g2d);
	else
	    g2d.fill(this);
    }

    /**
     * Draw the image associated with the IMAGE_ATTR
     * using the supplied Graphics2D context.
     *
     * @param g2d the Graphics2D context to be used for drawing
     */
    void drawImage(java.awt.Graphics2D g2d) {
	if(Grappa.waitForImages && imageLoading) {
	    synchronized(this) {
		try {
		    wait();
		}
		catch(InterruptedException e) {}
	    }
	}
	if(image != null) {
	    if(shape instanceof CustomRenderer) {
		((CustomRenderer)shape).drawImage(g2d);
	    } else {
		Rectangle sbox = shape.getBounds();
		Shape clip = g2d.getClip();
		g2d.clip(shape);
		g2d.drawImage(image, sbox.x, sbox.y, sbox.width, sbox.height, null);
		g2d.setClip(clip);
	    }
	}
    }
}
