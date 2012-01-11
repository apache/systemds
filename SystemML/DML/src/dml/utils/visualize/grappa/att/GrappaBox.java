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
 * This class extends java.awt.geom.Rectangle2D.Double and provides built-in
 * string-to-Rectangle2D and Rectangle2D-to-string conversions suitable for Grappa.
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public class GrappaBox extends java.awt.geom.Rectangle2D.Double
{
    private boolean dimensioned = true;

    /**
     * Constructs and initializes a <code>GrappaBox</code> with
     * upper-left coordinates (0,&nbsp;0) and zero width and height.
     */
    public GrappaBox() {
    }

    /**
     * Constructs and initializes a <code>GrappaBox</code> from a
     * Rectangle2D
     * @param r the rectangle defining the box (1-to-1 correspondence)
     */
    public GrappaBox(java.awt.geom.Rectangle2D r) {
	this(r.getX(),r.getY(),r.getWidth(),r.getHeight());
    }

    /**
     * Constructs and initializes a <code>GrappaBox</code> with the
     * specified coordinates.
     * @param x,&nbsp;y the upper-left position coordinates of the box
     * @param width,&nbsp;height the size of the box
     */
    public GrappaBox(double x, double y, double width, double height) {
	this.x = x;
	this.y = y;
	this.width = width;
	this.height = height;
    }

    /**
     * Constructs and initializes a <code>GrappaBox</code> with the
     * coordinates derived from the specified String representation.
     * When the <I>dimensioned</I> parameter is true, then the String
     * format should be:
     *    "<I>x-coord</I>,<I>y-coord</I>,<I>width</I>,<I>height</I>"
     * otherwise it should be:
     *    "<I>x1-coord</I>,<I>y1-coord</I>,<I>x2-coord</I>,<I>y2-coord</I>"
     * @param coordString String representing the coordinates to which to
     * set the newly constructed <code>GrappaBox</code>
     * @param dimensioned a boolean indicating the format of the string
     */
    public GrappaBox(String coordString, boolean dimensioned) {
	this.dimensioned = dimensioned;

	double[] coords = null;
	try {
	    coords = GrappaSupport.arrayForTuple(coordString);
	}
	catch(NumberFormatException nfe) {
	    throw new IllegalArgumentException("coordinate string (" + coordString + ") has a bad number format (" + nfe.getMessage() + ")");
	}
	if(coords == null || coords.length != 4) {
	    throw new IllegalArgumentException("coordinate string (" + coordString + ") does not contain 4 valid coordinates");
	}
	if(dimensioned) { // x1, y2, width, height
	    this.x = coords[0];
	    this.y = (Grappa.negateStringYCoord?-coords[1]:coords[1]);
	    this.width = coords[2];
	    this.height = coords[3];
	} else { // x1, y1, x2, y2
	    double tmp;
	    if(Grappa.negateStringYCoord) {
		coords[1] = -coords[1];
		coords[3] = -coords[3];
	    }
	    if(coords[0] > coords[2]) {
		tmp = coords[0];
		coords[0] = coords[2];
		coords[2] = tmp;
	    }
	    if(coords[1] > coords[3]) {
		tmp = coords[1];
		coords[1] = coords[3];
		coords[3] = tmp;
	    }
	    this.x = coords[0];
	    this.y = coords[1];
	    this.width = coords[2] - coords[0];
	    this.height = coords[3] - coords[1];
	}
    }

    /**
     * Constructs and initializes a <code>GrappaBox</code> with the
     * coordinates derived from the specified String representation.
     * The String format should be: "<I>x-coord</I>,<I>y-coord</I>,<I>width</I>,<I>height</I>""
     * @param coordString String representing the coordinates to which to
     * set the newly constructed <code>GrappaBox</code>
     */
    public GrappaBox(String coordString) {
	this(coordString, true);
    }

    /**
     * Provides a string representation of this object consistent 
     * with Grappa attributes.
     *
     * @return attribute-suitable string representation of this GrappaBox.
     */
    public String toAttributeString() {
	return(toFormattedString("%b"));
    }

    /**
     * Provides a formatted string representation of this object.
     * 
     * @param format the format used to build the string (<TT>%b</TT> is the base directive for a GrappaBox).
     * @return a string representation of this GrappaBox. 
     */
    public String toFormattedString(String format) {
	return(GrappaSupportPrintf.sprintf(new Object[] { format, this }));
    }

    /**
     * Provides a generic string representation of this object.
     * 
     * @return a generic string representation of this GrappaBox. 
     */
    public String toString() {
	return(x+","+y+","+width+","+height);
    }

    /**
     * Returns true if the String format will be "x1,x2,width,height"
     * or false if the format will be "x1,y1,x2,y2". The value is
     * determined at creation time.
     * 
     * @return a boolean indicating the format of this Object's string representation
     */
    public boolean isDimensioned() {
	return(dimensioned);
    }
}
