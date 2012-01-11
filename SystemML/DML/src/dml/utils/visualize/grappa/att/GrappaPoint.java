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
 * This class extends java.awt.geom.Point2D.Double and provides built-in
 * string-to-Point2D and Point2D-to-string conversions suitable for Grappa.
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public class GrappaPoint extends java.awt.geom.Point2D.Double
{
    /**
     * Constructs and initializes a <code>GrappaPoint</code> with
     * coordinates (0,&nbsp;0).
     */
    public GrappaPoint() {
    }

    /**
     * Constructs and initializes a <code>GrappaPoint</code> with the
     * specified coordinates.
     * @param x,&nbsp;y the coordinates to which to set the newly
     * constructed <code>GrappaPoint</code>
     */
    public GrappaPoint(double x, double y) {
	this.x = x;
	this.y = y;
    }

    /**
     * Constructs and initializes a <code>GrappaPoint</code> with the
     * coordinates derived from the specified String representation.
     * The String format should be: "<I>x-coord</I>,<I>y-coord</I>"
     * @param coordString String representing the coordinates to which to
     * set the newly constructed <code>GrappaPoint</code>
     */
    public GrappaPoint(String coordString) {
	double[] coords = null;
	try {
	    coords = GrappaSupport.arrayForTuple(coordString);
	}
	catch(NumberFormatException nfe) {
	    throw new IllegalArgumentException("coordinate string (" + coordString + ") has a bad number format (" + nfe.getMessage() + ")");
	}
	if(coords == null || coords.length != 2) {
	    throw new IllegalArgumentException("coordinate string (" + coordString + ") does not contain 2 valid coordinates");
	}
	this.x = coords[0];
	this.y = (Grappa.negateStringYCoord?-coords[1]:coords[1]);
    }

    /**
     * Provides a string representation of this object consistent 
     * with Grappa attributes.
     *
     * @return attribute-suitable string representation of this GrappaPoint.
     */
    public String toAttributeString() {
	return(toFormattedString("%p"));
    }

    /**
     * Provides a formatted string representation of this object.
     * 
     * @param format the format used to build the string (<TT>%p</TT> is the base directive for a GrappaPoint).
     * @return a string representation of this GrappaPoint. 
     */
    public String toFormattedString(String format) {
	return(GrappaSupportPrintf.sprintf(new Object[] { format, this }));
    }

    /**
     * Provides a generic string representation of this object.
     * 
     * @return a generic string representation of this GrappaPoint. 
     */
    public String toString() {
	return(x+","+y);
    }
}
