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
import java.awt.geom.*;

/**
 * This class provides line and bezier-curve support for Grappa.
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public class GrappaLine
    implements
	GrappaConstants,
	Cloneable, Shape
{
    /**
     * Arrow head length
     */
    public final static double arrowLength     = 10;

    /**
     * Arrow head width
     */
    public final static double arrowWidth      = 5;

    /**
     * Bit flag to indicate that line has no arrow heads.
     */
    public static final int NONE_ARROW_EDGE		= 0;
    /**
     * Bit flag to indicate that line has an arrow head at its head end.
     */
    public static final int HEAD_ARROW_EDGE		= 1;
    /**
     * Bit flag to indicate that line has an arrow head at its tail end.
     */
    public static final int TAIL_ARROW_EDGE		= 2;
    /**
     * Bit flag to indicate that line has arrow heads at both ends.
     * Note that
     * NONE_ARROW_EDGE + HEAD_ARROW_EDGE + TAIL_ARROW_EDGE = BOTH_ARROW_EDGE
     */
    public static final int BOTH_ARROW_EDGE		= 3;

    // the general path describing this line (including arrow heads)
    private GeneralPath path = null;
    // fatter path for contains and intersects tests
    private GeneralPath testpath = null;
    // arrow head info
    private int arrow = NONE_ARROW_EDGE;
    // the point set for this line (not including arrow heads)
    private GrappaPoint[] gpts = null;

    // fix winding rule at instantiation time
    private int windingRule = Grappa.windingRule;

    ////////////////////////////////////////////////////////////////////////
    //
    // Constructors
    //
    ////////////////////////////////////////////////////////////////////////

    /**
     * Constructs a new <code>GrappaLine</code> object from an array of 
     * (cubic) curve points.
     * The winding rule for this path is defaulted (from Grappa.windingRule).
     * @param pts the <code>GrappaPoint</code> array used to describe the curve
     * @param type indicates arrow type (NONE_ARROW_EDGE,HEAD_ARROW_EDGE,
     *             TAIL_ARROW_EDGE,BOTH_ARROW_EDGE)
     */
    public GrappaLine(GrappaPoint[] pts, int type) {
	updateLine(pts,type);
    }

    /**
     * Constructs a new <code>GrappaLine</code> object from a string of 
     * (cubic) curve points as used by "dot".
     * All of the initial geometry and the winding rule for this path are
     * defaulted.
     * @param curve the <code>String</code> that specifies the point list; the
     *              format is: [s,x0,y0|e,xN,yN] [x1,y2] ... [xN-1,yN-1]
     */
    public GrappaLine(String curve) {
	updateLine(curve);
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // Public methods
    //
    ////////////////////////////////////////////////////////////////////////

    /**
     * Check for equality of this object with the supplied object.
     *
     * @param the object to be checked for equality
     * @return true, when equal
     */
    public boolean equals(Object obj) {
	if(obj == null || !(obj instanceof GrappaLine)) return(false);
	GrappaLine cmp = (GrappaLine)obj;
	if(cmp == this) return(true);
	if(cmp.getArrowType() != arrow) return(false);
	if(cmp.gpts.length != gpts.length || !gpts.equals(cmp.gpts)) return(false);
	// should be sufficient, should be no need to compare path
	//if(!path.equals(cmp.path)) return(false);
	return(true);
    }

    /**
     * Return the arrow type for this line.
     * @return one of NONE_ARROW_EDGE,HEAD_ARROW_EDGE, TAIL_ARROW_EDGE, or BOTH_ARROW_EDGE
     */
    public int getArrowType() {
	return arrow;
    }

    /**
     * Return the winding rule for this line.
     * @return one of WIND_NON_ZERO or WIND_EVEN_ODD
     */
    public int getWindingRule() {
	return windingRule;
    }

    /**
     * Check is the line is oriented away from the given point. 
     *
     * @return true if the line is oriented so that its starting point is
     *  nearer to the supplied point than its ending point.
     */
    public boolean startsNear(Point2D pt) {
	return(gpts[0].distance(pt) < gpts[gpts.length-1].distance(pt));
    }

    /**
     * Provides a string representation of this object consistent 
     * with Grappa attributes.
     *
     * @return attribute-suitable string representation of this GrappaLine.
     */
    public String toAttributeString() {
	return(toFormattedString("%p"));
    }

    /**
     * Provides a formatted string representation of this object.
     * 
     * @param pointFormat the specific format directive to use for each point in the line (<TT>%p</TT> is the base directive).
     * @return a string representation of this GrappaLine. 
     */
    public String toFormattedString(String pointFormat) {
	int ps = 0;
	int pe = gpts.length - 1;
	boolean spacer = false;
	StringBuffer buf = new StringBuffer();
	if((arrow&HEAD_ARROW_EDGE) != 0) {
	    buf.append("s,");
	    buf.append(gpts[ps++].toFormattedString(pointFormat));
	    spacer = true;
	}
	if((arrow&TAIL_ARROW_EDGE) != 0) {
	    if(spacer) {
		buf.append(" e,");
	    } else {
		buf.append("e,");
		spacer = true;
	    }
	    buf.append(gpts[pe--].toFormattedString(pointFormat));
	}
	while(ps <= pe) {
	    if(spacer) {
		buf.append(" ");
	    } else {
		spacer = true;
	    }
	    buf.append(gpts[ps++].toFormattedString(pointFormat));
	}
	return(buf.toString());
    }

    /**
     * Provides a generic string representation of this object.
     * 
     * @return a generic string representation of this GrappaLine. 
     */
    public String toString() {
	int ps = 0;
	int pe = gpts.length - 1;
	boolean spacer = false;
	StringBuffer buf = new StringBuffer();
	if((arrow&HEAD_ARROW_EDGE) != 0) {
	    buf.append("s,");
	    buf.append(gpts[ps].x);
	    buf.append(",");
	    buf.append(gpts[ps++].y);
	    spacer = true;
	}
	if((arrow&TAIL_ARROW_EDGE) != 0) {
	    if(spacer) {
		buf.append(" e,");
	    } else {
		buf.append("e,");
		spacer = true;
	    }
	    buf.append(gpts[pe].x);
	    buf.append(",");
	    buf.append(gpts[pe--].y);
	}
	while(ps <= pe) {
	    if(spacer) {
		buf.append(" ");
	    } else {
		spacer = true;
	    }
	    buf.append(gpts[ps].x);
	    buf.append(",");
	    buf.append(gpts[ps++].y);
	}
	return(buf.toString());
    }

    /**
     * Changes the arrow type for this line.
     *
     * @param new_type indicates arrow type (NONE_ARROW_EDGE,HEAD_ARROW_EDGE,
     *                 TAIL_ARROW_EDGE,BOTH_ARROW_EDGE)
     * 
     * @return true if the type changed, false otherwise.
     */
    public boolean changeArrowType(int new_type) {
	boolean changed = false;
	if(arrow != new_type && (new_type&(~(BOTH_ARROW_EDGE))) == 0) {
	    changed = true;
	    updateLine(gpts, new_type);
	}
	return(changed);
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // Private methods
    //
    ////////////////////////////////////////////////////////////////////////

    // add an arrow to the path of this line
    private void addArrow(GeneralPath path, GeneralPath testpath, GrappaPoint tip, GrappaPoint shaft, double length, double width) {
	double theta = Math.atan2((tip.y - shaft.y), (tip.x - shaft.x));
	double half_width = width / 2.0;

	float x, y;

	path.lineTo(
		    x = (float) (tip.x - (length * Math.cos(theta) - half_width * Math.sin(theta))),
		    y = (float) (tip.y - (length * Math.sin(theta) + half_width * Math.cos(theta)))
		    );
	testpath.lineTo(x,y);
	path.lineTo(
		    x = (float) (tip.x - (length * Math.cos(theta) + half_width * Math.sin(theta))),
		    y = (float) (tip.y - (length * Math.sin(theta) - half_width * Math.cos(theta)))
		    );
	testpath.lineTo(x,y);
	path.lineTo(
		    x = (float) tip.x,
		    y = (float) tip.y
		    );
	testpath.lineTo(x,y);
    } 

    // translate the supplied string into the points of this line
    private void updateLine(String curve) {
	int type = NONE_ARROW_EDGE;
	int i, j, k;
	int len = curve.length();
	int pts = 1;
	boolean wasSpace = true;
	GrappaPoint[] grpts = null;
	Integer attr_type;

	// first pass is mostly sizing and basic validity check
	for(i = 0, j = len; i < len; i++) {
	    switch((int)curve.charAt(i)) {
	    case 's':
		wasSpace = false;
		type += HEAD_ARROW_EDGE;
		break;
	    case 'e':
		wasSpace = false;
		type += TAIL_ARROW_EDGE;
		break;
	    case ' ':
		if(!wasSpace) {
		    if(j == len) j = i; // first space (used later)
		    pts++;
		    wasSpace = true;
		}
		break;
	    default:
		wasSpace = false;
		break;
	    }
	}
	if(wasSpace) pts--;
	if(pts < 2 || type > BOTH_ARROW_EDGE) {
	    throw new IllegalArgumentException("bad curve specifier string (" + curve + ")");
	}

	grpts = new GrappaPoint[pts];

	for(i = 0; i < len; i++) {
	    if(curve.charAt(i) != ' ') {
		break;
	    }
	}
	pts = 0;
	if(curve.charAt(i) == 's') {
	    grpts[pts++] = new GrappaPoint(curve.substring(i+2,j));
	    for(i = ++j; i < len; i++) {
		if(curve.charAt(i) != ' ') {
		    break;
		}
	    }
	    for(k = j, j = len; k < j; k++) {
		if(curve.charAt(k) == ' ') {
		    j = k;
		    break;
		}
	    }
	}
	if(curve.charAt(i) == 'e') {
	    grpts[grpts.length-1] = new GrappaPoint(curve.substring(i+2,j));
	    for(i = ++j; i < len; i++) {
		if(curve.charAt(i) != ' ') {
		    break;
		}
	    }
	    for(k = j, j = len; k < j; k++) {
		if(curve.charAt(k) == ' ') {
		    j = k;
		    break;
		}
	    }
	}
	if(curve.charAt(i) == 's') {
	    grpts[pts++] = new GrappaPoint(curve.substring(i+2,j));
	    for(i = ++j; i < len; i++) {
		if(curve.charAt(i) != ' ') {
		    break;
		}
	    }
	    for(k = j, j = len; k < j; k++) {
		if(curve.charAt(k) == ' ') {
		    j = k;
		    break;
		}
	    }
	}
	while(i < len) {
	    grpts[pts++] = new GrappaPoint(curve.substring(i,j));
	    for(i = ++j; i < len; i++) {
		if(curve.charAt(i) != ' ') {
		    break;
		}
	    }
	    for(k = j, j = len; k < j; k++) {
		if(curve.charAt(k) == ' ') {
		    j = k;
		    break;
		}
	    }
	}

	updateLine(grpts, type);
    }

    // given points and an arrow type, generate the path of this line
    private void updateLine(GrappaPoint[] grpts, int type) {
	int pts = 0;
	int xpts = 0;

	float x, y, x2, y2, x3, y3, z = -2;

	if((type&HEAD_ARROW_EDGE) != 0) xpts += 3;
	if((type&TAIL_ARROW_EDGE) != 0) xpts += 3;

	GeneralPath grpath = new GeneralPath(windingRule, grpts.length+xpts+grpts.length-1);
	GeneralPath grtestpath = new GeneralPath(windingRule, grpts.length+xpts+grpts.length-1);

	if(grpts.length < 2) {
	    throw new IllegalArgumentException("need at least two supplied points");
	}

	grpath.moveTo(x = (float)grpts[pts].x, y = (float)grpts[pts++].y);
	grtestpath.moveTo(x+z,y+z);
	if((type&HEAD_ARROW_EDGE) != 0) {
	    grtestpath.moveTo(x+z,y+z);
	    addArrow(grpath, grtestpath, grpts[pts-1], grpts[pts], arrowLength, arrowWidth);
	    grpath.lineTo(x = (float)grpts[pts].x, y = (float)grpts[pts++].y);
	    grtestpath.lineTo(x-z, y-z);
	} else grtestpath.lineTo(x-z,y-z);
	boolean lastWasLine = false;
	while(pts < grpts.length) {
	    lastWasLine = false;
	    if(pts+3 <= grpts.length) {
		grpath.curveTo(
			       x = (float)grpts[pts].x, y = (float)grpts[pts++].y,
			       x2 = (float)grpts[pts].x, y2 = (float)grpts[pts++].y,
			       x3 = (float)grpts[pts].x, y3 = (float)grpts[pts++].y
			       );
		grtestpath.curveTo(x-z,y-z,x2-z,y2-z,x3-z,y3-z);
	    } else {
		lastWasLine = true;
		grpath.lineTo(x = (float)grpts[pts].x, y = (float)grpts[pts++].y);
		grtestpath.lineTo(x-z,y-z);
	    }
	}
	if((type&TAIL_ARROW_EDGE) != 0) {
	    addArrow(grpath, grtestpath,  grpts[pts-1], grpts[pts-2], arrowLength, arrowWidth);
	}
	pts--;
	while(pts > 0) {
	    if(!lastWasLine && pts-3 >= 0) {
		grpath.curveTo(
			       x = (float)grpts[--pts].x, y = (float)grpts[pts].y,
			       x2 = (float)grpts[--pts].x, y2 = (float)grpts[pts].y,
			       x3 = (float)grpts[--pts].x, y3 = (float)grpts[pts].y
			       );
		grtestpath.curveTo(x+z,y+z,x2+z,y2+z,x3+z,y3+z);
	    } else {
		lastWasLine = false;
		grpath.lineTo(x = (float)grpts[--pts].x, y = (float)grpts[pts].y);
		grtestpath.lineTo(x+z,y+z);
	    }
	}

	this.gpts = grpts;
	this.path = grpath;
	this.testpath = grtestpath;
	this.arrow = type;
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
	    GrappaLine copy = (GrappaLine) super.clone();
	    copy.path = (GeneralPath) path.clone();
	    if(gpts != null) {
		copy.gpts = (GrappaPoint[])(gpts.clone());
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

    public final boolean contains(double x, double y) {
	return(testpath.contains(x, y));
    }

    public final boolean contains(double x, double y, double width, double height) {
	return(testpath.contains(x, y, width, height));
    }

    public final boolean contains(Point2D p) {
	return(testpath.contains(p));
    }

    public final boolean contains(Rectangle2D r) {
	return(testpath.contains(r));
    }

    public final Rectangle getBounds() {
	return(path.getBounds2D().getBounds());
    }

    public final Rectangle2D getBounds2D() {
	return(path.getBounds2D());
    }

    /**
     * Equivalent to <TT>getPathIterator(null)</TT>.
     *
     * @see getPathIterator(AffineTransform)
     */
    public final PathIterator getPathIterator() {
	return path.getPathIterator(null);
    }

    public final PathIterator getPathIterator(AffineTransform at) {
	return path.getPathIterator(at);
    }

    public final PathIterator getPathIterator(AffineTransform at, double flatness) {
	return new FlatteningPathIterator(path.getPathIterator(at), flatness);
    }

    public final boolean intersects(double x, double y, double width, double height) {
	return(testpath.intersects(x, y, width, height));
    }

    public final boolean intersects(Rectangle2D r) {
	return(testpath.intersects(r));
    }
}
