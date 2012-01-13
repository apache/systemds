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
 * This class provides a flexible, parameterized polygonal shape builder.
 * The guts of a GrappaShape is a GeneralPath object.
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public class GrappaShape
    implements
	GrappaConstants,
	Cloneable, Shape
{

    /**
     * path that defines shape
     */
    protected GeneralPath path = null;

    private final static double RBCONST = 12.0;
    private final static double RBCURVE = 0.5;

    private final static double CIRCLE_XDIAG = Math.sqrt(7.0/16.0);
    private final static double CIRCLE_YDIAG = 0.75;


    ////////////////////////////////////////////////////////////////////////
    //
    // Constructors
    //
    ////////////////////////////////////////////////////////////////////////

    /**
     * Constructs a new <code>GrappaShape</code> object.
     * The winding rule for this path is defaulted (from Grappa.windingRule).
     *
     * @param type the shape specifier (.e.g., EGG_SHAPE).
     * @param x the x-coordinate of the polygon center point.
     * @param y the y-coordinate of the polygon center point.
     * @param width the overall width of the polygon bounding box.
     * @param height the overall height of the polygon bounding box.
     * @param sidesArg the number of sides of the polygon.
     * @param peripheriesArg the number of peripheries (outlines) of the polygon.
     * @param distortionArg a distortion factor applied to the polygon shape, when non-zero
     * @param skewArg a skewing factor applied to the polygon shape, when non-zero
     * @param orientationArg an orientation angle in radians.
     * @param roundedArg a directive to round the corners.
     * @param diagonalsArg a directive to draw catercorners at the vertices.
     * @param extra used only with the <I>record</I> shape at present, it provides a string of tuples giving interior partition information.
     */
    public GrappaShape(int type, double x, double y, double width, double height, int sidesArg, int peripheriesArg, double distortionArg, double skewArg, double orientationArg, boolean roundedArg, boolean diagonalsArg, Object extra) {

	path = new GeneralPath(Grappa.windingRule);

	// defaults
	int sides = 120;
	int peripheries = 1;
	double distortion = 0;
	double skew = 0;
	double orientation = 0;
	boolean rounded = false;
	boolean diagonals = false;
	float[] rects = null;
	Point2D.Float rectsMin = null;
	Point2D.Float rectsMax = null;
	int i;

	type &= SHAPE_MASK;

	switch(type) {
	case MSQUARE_SHAPE:
	    diagonals = true;
	    type = BOX_SHAPE;
	    // fall through to BOX_SHAPE case
	case BOX_SHAPE:
	    sides = 4;
	    break;
	case MDIAMOND_SHAPE:
	    diagonals = true;
	    type = DIAMOND_SHAPE;
	    // fall through to DIAMOND_SHAPE case
	case DIAMOND_SHAPE:
	    sides = 4;
	    orientation = Math.PI / 4.0;
	    break;
	case DOUBLECIRCLE_SHAPE:
	    peripheries = 2;
	    break;
	case DOUBLEOCTAGON_SHAPE:
	    sides = 8;
	    peripheries = 2;
	    break;
	case EGG_SHAPE:
	    distortion = -0.3;
	    break;
	case HEXAGON_SHAPE:
	    sides = 6;
	    break;
	case HOUSE_SHAPE:
	    sides = 5;
	    distortion = -0.64;
	    break;
	case INVERTEDHOUSE_SHAPE:
	    sides = 5;
	    distortion = -0.64;
	    orientation = Math.PI;
	    break;
	case INVERTEDTRAPEZIUM_SHAPE:
	    sides = 4;
	    distortion = -0.4;
	    orientation = Math.PI;
	    break;
	case INVERTEDTRIANGLE_SHAPE:
	    sides = 3;
	    orientation = Math.PI;
	    break;
	case MRECORD_SHAPE:
	    rounded = true;
	    type = RECORD_SHAPE;
	    // fall through to RECORD_SHAPE case
	case RECORD_SHAPE:
	    sides = 4;
	    break;
	case PARALLELOGRAM_SHAPE:
	    sides = 4;
	    skew = 0.6;
	    break;
	case PENTAGON_SHAPE:
	    sides = 5;
	    break;
	case OCTAGON_SHAPE:
	    sides = 8;
	    break;
	case PLAINTEXT_SHAPE:
	    sides = 0;
	    break;
	case ROUNDEDBOX_SHAPE:
	    sides = 4;
	    rounded = true;
	    break;
	case TRAPEZIUM_SHAPE:
	    sides = 4;
	    distortion = -0.4;
	    break;
	case TRIANGLE_SHAPE:
	    sides = 3;
	    break;
	case TRIPLEOCTAGON_SHAPE:
	    sides = 8;
	    peripheries = 3;
	    break;
	case MCIRCLE_SHAPE:
	    diagonals = true;
	    type = OVAL_SHAPE;
	    // fall through to OVAL_SHAPE case
	case OVAL_SHAPE:
	case POINT_SHAPE:
	case POLYGON_SHAPE:
	default:
	    break;
	case CUSTOM_SHAPE:
	    sidesArg = 4;
	    peripheriesArg = 0;
	    distortionArg = 0;
	    skewArg = 0;
	    orientationArg = 0;
	    roundedArg = false;
	    diagonalsArg = false;
	    extra = null;
	    break;
	}

	sides = (sidesArg >= 0) ? sidesArg : sides;
	peripheries = (peripheriesArg >= 0) ? peripheriesArg : peripheries;
	distortion = (distortionArg != 0) ? distortionArg : distortion;
	skew = (skewArg != 0) ? skewArg : skew;
	orientation = (orientationArg != 0) ? orientationArg : orientation;
	rounded = (roundedArg) ? roundedArg : rounded;
	diagonals = (diagonalsArg) ? diagonalsArg : diagonals;

	if(sides < 3) {
	    sides = 0;
	}

	if(type == OVAL_SHAPE || type == POINT_SHAPE) {
	    sides = 120;
	} else if(type == RECORD_SHAPE) {
	    // basically, only rounded is an option
	    sides = 4;
	    peripheries = 1;
	    distortion = 0;
	    skew = 0;
	    orientation = 0;
	    diagonals = false;
	    if(extra != null && extra instanceof String) {
		rects = GrappaSupport.floatArrayForTuple((String)extra);
		if((rects.length % 4) == 0) {
		    rectsMin = new Point2D.Float();
		    rectsMax = new Point2D.Float();
		    rectsMax.x = rectsMin.x = rects[0];
		    if(Grappa.negateStringYCoord) {
			rectsMax.y = rectsMin.y = -rects[1];
		    } else {
			rectsMax.y = rectsMin.y = rects[1];
		    }
		    for(i = 0; i < rects.length; i += 2) {
			if(Grappa.negateStringYCoord)
			    rects[i+1] = -rects[i+1];
			if(rects[i] < rectsMin.x)
			    rectsMin.x = rects[i];
			if(rects[i] > rectsMax.x)
			    rectsMax.x = rects[i];
			if(rects[i+1] < rectsMin.y)
			    rectsMin.y = rects[i+1];
			if(rects[i+1] > rectsMax.y)
			    rectsMax.y = rects[i+1];
		    }
		    width = rectsMax.x - rectsMin.x;
		    height = rectsMax.y - rectsMin.y;
		    x = rectsMin.x + (width / 2.0);
		    y = rectsMin.y + (height / 2.0);
		} else {
		    rects = null;
		}
	    }
	}

	if(peripheries < 1 || sides == 0) {
	    path.moveTo((float)x,(float)y);
	    return;
	}

	double dtmp, alpha, beta;

	double sectorAngle = 2.0 * Math.PI / (double)sides;
	double sideLength  = Math.sin(sectorAngle/2.0);

	double skewDist, gDist, gSkew, angle;
	if(skew == 0 && distortion == 0) {
	    skewDist = 1;
	    gDist = 0;
	    gSkew = 0;
	} else {
	    skewDist = Math.abs(distortion) + Math.abs(skew);
	    skewDist = Math.sqrt(skewDist * skewDist + 1.0);
	    gDist = distortion * Math.sqrt(2.0) / Math.cos(sectorAngle/2.0);
	    gSkew = skew / 2.0;
	}
	GrappaPoint Rpt = new GrappaPoint();
	GrappaPoint Ppt = new GrappaPoint();
	GrappaPoint Bpt = new GrappaPoint();
	int verts = 0;
	double sinX, cosX;

	GrappaPoint[] rawVertices = new GrappaPoint[sides];
	GrappaPoint[] tmpVertices = null;
	if(diagonals) {
	    tmpVertices = new GrappaPoint[2 * sides];
	} else if(rounded) {
	    tmpVertices = new GrappaPoint[4 * sides];
	}

	GrappaPoint Pt0;
	GrappaPoint Pt1;
	double tmp1 = 0;
	double tmp2 = 0;
	int tverts = 0;

	double delta_w = 0, delta_h = 0;
	double minX = 0, minY = 0, maxX = 0, maxY = 0;

	boolean tooSmall = false;

	tmp1 = (peripheries - 1) * (2 * PERIPHERY_GAP);

	if((width - tmp1) <= 0 || (height - tmp2) <= 0) {
	    peripheries = 1;
	}

	for(int j = 0; j < peripheries; j++) {

	    if(j > 0) {
		if(j == 1) {

		    Ppt.x = rawVertices[sides-1].x;
		    Ppt.y = rawVertices[sides-1].y;

		    Rpt.x = rawVertices[0].x;
		    Rpt.y = rawVertices[0].y;

		    beta = Math.atan2(Rpt.y-Ppt.y,Rpt.x-Ppt.x);

		    for(i = 0; i < sides; i++) {
			Ppt.x = Rpt.x;
			Ppt.y = Rpt.y;
			Rpt.x = rawVertices[(i+1)%sides].x;
			Rpt.y = rawVertices[(i+1)%sides].y;
			alpha = beta;
			beta = Math.atan2(Rpt.y-Ppt.y,Rpt.x-Ppt.x);
			tmp1 = (alpha + Math.PI - beta) / 2.0;

			/*
			 * find the distance along the bisector to the
			 * intersection of the next periphery
			 */
			dtmp = PERIPHERY_GAP / Math.sin(tmp1);

			// convert this distance to x and y
			tmp2 = alpha - tmp1;
			sinX = Math.sin(tmp2);
			cosX = Math.cos(tmp2);
			sinX *= dtmp;
			cosX *= dtmp;

			tmp1 = Ppt.x - cosX;
			tmp2 = Ppt.y - sinX;

			if(i == 0) {
			    maxX = minX = tmp1;
			    maxY = minY = tmp2;
			} else {
			    if(minX > tmp1) minX = tmp1;
			    if(maxX < tmp1) maxX = tmp1;
			    if(minY > tmp2) minY = tmp2;
			    if(maxY < tmp2) maxY = tmp2;
			}
		    }
		    delta_w = width - (maxX - minX);
		    delta_h = height - (maxY - minY);
		}
		width -= delta_w;
		height -= delta_h;
	    }

	    angle = (sectorAngle - Math.PI)/2.0;
	    //angle = sectorAngle/2.0;
	    sinX = Math.sin(angle);
	    cosX = Math.cos(angle);
	    Rpt.x = 0.5 * cosX;
	    Rpt.y = 0.5 * sinX;
	    angle += (Math.PI - sectorAngle)/2.0;
	    //angle = Math.PI/2.0;

	    Bpt.x = 0;
	    Bpt.y = 0;

	    verts = 0;
	    for(i = 0; i < sides; i++) {
		// next regular vertex
		angle += sectorAngle;
		sinX = Math.sin(angle);
		cosX = Math.cos(angle);
		Rpt.x += sideLength*cosX;
		Rpt.y += sideLength*sinX;
		// distort and skew
		Ppt.x = Rpt.x * (skewDist + Rpt.y * gDist) + Rpt.y * gSkew;
		Ppt.y = Rpt.y;
		// orient Ppt
		if(orientation != 0) {
		    alpha = orientation + Math.atan2(Ppt.y,Ppt.x); 
		    sinX = Math.sin(alpha);
		    cosX = Math.cos(alpha);
		    dtmp = Ppt.distance(0,0);
		    Ppt.x = dtmp * cosX;
		    Ppt.y = dtmp * sinX;
		}
		// scale
		Ppt.x *= width;
		Ppt.y *= height;
		// store result
		rawVertices[verts++] = (GrappaPoint)Ppt.clone();
		if(Bpt.x < Math.abs(Ppt.x)) Bpt.x = Math.abs(Ppt.x);
		if(Bpt.y < Math.abs(Ppt.y)) Bpt.y = Math.abs(Ppt.y);
	    }

	    Bpt.x = width  / (2.0 * Bpt.x);
	    Bpt.y = height / (2.0 * Bpt.y);

	    for(i = 0; i < sides; i++) {
		rawVertices[i].x *= Bpt.x;
		rawVertices[i].y *= Bpt.y;
	    }


	    if((rounded || diagonals) && type != OVAL_SHAPE && type != POINT_SHAPE && j == (peripheries - 1)) {
		tooSmall = false;
		tverts = 0;
		Pt0 = rawVertices[0];
		for(i = 0; i < sides; i++) {
		    // already scaled
		    Pt0 = rawVertices[i];
		    if(i < sides - 1) {
			Pt1 = rawVertices[i+1];
		    } else {
			Pt1 = rawVertices[0];
		    }
		    tmp2 = Pt0.distance(Pt1);
		    if(tmp2 < RBCONST) {
			tooSmall = true;
			break;
		    }
		    tmp1 = RBCONST / tmp2;
		    if(!diagonals) {
			tmp2 = RBCURVE * tmp1;
			tmpVertices[tverts++] = new GrappaPoint(
								Pt0.x + tmp2 * (Pt1.x - Pt0.x),
								Pt0.y + tmp2 * (Pt1.y - Pt0.y)
								);
		    }
		    tmpVertices[tverts++] = new GrappaPoint(
							    Pt0.x + tmp1 * (Pt1.x - Pt0.x),
							    Pt0.y + tmp1 * (Pt1.y - Pt0.y)
							    );
		    tmp1 = 1 - tmp1;
		    tmpVertices[tverts++] = new GrappaPoint(
							    Pt0.x + tmp1 * (Pt1.x - Pt0.x),
							    Pt0.y + tmp1 * (Pt1.y - Pt0.y)
							    );
		    if(!diagonals) {
			tmp2 = 1 - tmp2;
			tmpVertices[tverts++] = new GrappaPoint(
								Pt0.x + tmp2 * (Pt1.x - Pt0.x),
								Pt0.y + tmp2 * (Pt1.y - Pt0.y)
								);
		    }
		}

		if(tooSmall) {
		    for(i = 0; i < sides; i++) {
			if(i == 0) {
			    path.moveTo((float)(x + rawVertices[i].x), (float)(y - rawVertices[i].y));
			} else {
			    path.lineTo((float)(x + rawVertices[i].x), (float)(y - rawVertices[i].y));
			}
		    }
		} else {
		    if(diagonals) {
			path.moveTo((float)(x + tmpVertices[0].x), (float)(y - tmpVertices[0].y));
			for(i = (2*sides)-1; i > 0; i-=2) {
			    path.lineTo((float)(x + tmpVertices[i].x), (float)(y - tmpVertices[i].y));
			    path.moveTo((float)(x + tmpVertices[i-1].x), (float)(y - tmpVertices[i-1].y));
			}
			for(i = 0; i < sides; i++) {
			    if(i == 0) {
				path.moveTo((float)(x + rawVertices[i].x), (float)(y - rawVertices[i].y));
			    } else {
				path.lineTo((float)(x + rawVertices[i].x), (float)(y - rawVertices[i].y));
			    }
			}
		    } else {
			path.moveTo((float)(x + tmpVertices[2].x), (float)(y - tmpVertices[2].y));
			for(i = 3; i < (4*sides)-2; i+=4) {
			    path.curveTo(
					 (float)(x + tmpVertices[i].x), (float)(y - tmpVertices[i].y),
					 (float)(x + tmpVertices[i+1].x), (float)(y - tmpVertices[i+1].y),
					 (float)(x + tmpVertices[i+2].x), (float)(y - tmpVertices[i+2].y)
					 );
			    path.lineTo((float)(x + tmpVertices[i+3].x), (float)(y - tmpVertices[i+3].y));
			}
			i = (4 * sides) - 1;
			path.curveTo(
				     (float)(x + tmpVertices[i].x), (float)(y - tmpVertices[i].y),
				     (float)(x + tmpVertices[0].x), (float)(y - tmpVertices[0].y),
				     (float)(x + tmpVertices[1].x), (float)(y - tmpVertices[1].y)
				     );
		    }
		}
	    } else {
		for(i = 0; i < sides; i++) {
		    if(i == 0) {
			path.moveTo((float)(x + rawVertices[i].x), (float)(y - rawVertices[i].y));
		    } else {
			path.lineTo((float)(x + rawVertices[i].x), (float)(y - rawVertices[i].y));
		    }
		}
	    }
	    path.closePath();
	}

	// special cases

	if(type == OVAL_SHAPE && diagonals) {

	    Pt0 = new GrappaPoint(
				  width * CIRCLE_XDIAG / 2.0,
				  height * CIRCLE_YDIAG / 2.0
				  );
	    Ppt.x = x + Pt0.x;
	    Ppt.y = y - Pt0.y;
	    Rpt.x = Ppt.x - 2.0 * Pt0.x;
	    Rpt.y = Ppt.y;

	    path.moveTo((float)Ppt.x, (float)Ppt.y);
	    path.lineTo((float)Rpt.x, (float)Rpt.y);
	    
	    Ppt.y += (2.0 * Pt0.y) - 1.0;
	    Rpt.y = Ppt.y;

	    path.moveTo((float)Ppt.x, (float)Ppt.y);
	    path.lineTo((float)Rpt.x, (float)Rpt.y);
	    
	} else if(type == RECORD_SHAPE) {
	    if(rects != null) {
		float tmp;
		for(i = 0; i < rects.length; i += 4) {
		    if(rects[i] > rects[i+2]) {
			tmp = rects[i];
			rects[i] = rects[i+2];
			rects[i+2] = tmp;
		    }
		    if(rects[i+1] > rects[i+3]) {
			tmp = rects[i+1];
			rects[i+1] = rects[i+3];
			rects[i+3] = tmp;
		    }
		    if(!(rects[i] == rectsMin.x && rects[i+2] == rectsMax.x) || rects[i+1] != rectsMin.y) {
			path.moveTo(rects[i],rects[i+1]);
			path.lineTo(rects[i+2],rects[i+1]);
		    }
		    if(rects[i+2] != rectsMax.x || !(rects[i+1] == rectsMin.y && rects[i+3] == rectsMax.y)) {
			path.moveTo(rects[i+2],rects[i+1]);
			path.lineTo(rects[i+2],rects[i+3]);
		    }
		    if(!(rects[i] == rectsMin.x && rects[i+2] == rectsMax.x) || rects[i+3] != rectsMax.y) {
			path.moveTo(rects[i+2],rects[i+3]);
			path.lineTo(rects[i],rects[i+3]);
		    }
		    if(rects[i] != rectsMin.x || !(rects[i+1] == rectsMin.y && rects[i+3] == rectsMax.y)) {
			path.moveTo(rects[i],rects[i+3]);
			path.lineTo(rects[i],rects[i+1]);
		    }
		}
	    }
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
	    GrappaShape copy = (GrappaShape) super.clone();
	    copy.path = (GeneralPath) path.clone();
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
	return(path.contains(x, y));
    }

    public final boolean contains(double x, double y, double width, double height) {
	return(path.contains(x, y, width, height));
    }

    public final boolean contains(Point2D p) {
	return(path.contains(p));
    }

    public final boolean contains(Rectangle2D r) {
	return(path.contains(r));
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
	return(path.intersects(x, y, width, height));
    }

    public final boolean intersects(Rectangle2D r) {
	return(path.intersects(r));
    }
}
