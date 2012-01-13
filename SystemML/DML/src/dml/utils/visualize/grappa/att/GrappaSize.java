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
 * This class extends java.awt.geom.Dimension2D and provides built-in
 * string-to-Dimension2D and Dimension2D-to-string conversions suitable for Grappa.
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public class GrappaSize extends java.awt.geom.Dimension2D
{
    /**
     * The width of the Dimension.
     */
    public double width;

    /**
     * The height of the Dimension.
     */
    public double height;

    /**
     * Constructs and initializes a <code>GrappaSize</code> with
     * coordinates (0,&nbsp;0).
     */
    public GrappaSize() {
    }

    /**
     * Constructs and initializes a <code>GrappaSize</code> with the
     * specified coordinates.
     * @param width,&nbsp;height the coordinates to which to set the newly
     * constructed <code>GrappaSize</code>
     */
    public GrappaSize(double width, double height) {
	this.width = width;
	this.height = height;
    }

    /**
     * Constructs and initializes a <code>GrappaSize</code> with the
     * coordinates derived from the specified String representation.
     * The String format should be: "<I>width</I>,<I>height</I>"
     * @param dimenString String representing the dimensions to which to
     * set the newly constructed <code>GrappaSize</code>
     */
    public GrappaSize(String dimenString) {
	double[] coords = null;
	try {
	    coords = GrappaSupport.arrayForTuple(dimenString);
	}
	catch(NumberFormatException nfe) {
	    throw new IllegalArgumentException("coordinate string (" + dimenString + ") has a bad number format (" + nfe.getMessage() + ")");
	}
	if(coords == null || coords.length != 2) {
	    throw new IllegalArgumentException("coordinate string (" + dimenString + ") does not contain 2 valid coordinates");
	}
	this.width = coords[0];
	this.height = coords[1];
    }

    /**
     * Returns the width.
     */
    public double getWidth() {
	return width;
    }

    /**
     * Returns the height.
     */
    public double getHeight() {
	return height;
    }

    /**
     * Sets the width and height.
     */
    public void setSize(java.awt.geom.Dimension2D d) {
	setSize(d.getWidth(), d.getHeight());
    }

    /**
     * Sets the width and height.
     */
    public void setSize(double width, double height) {
	this.width = width;
	this.height = height;
    }

    /**
     * Provides a string representation of this object consistent 
     * with Grappa attributes.
     *
     * @return attribute-suitable string representation of this GrappaSize.
     */
    public String toAttributeString() {
	return(toFormattedString("%p"));
    }

    /**
     * Provides a formatted string representation of this object.
     * 
     * @param format the format used to build the string (<TT>%p</TT> is the base directive suitable for a GrappaSize).
     * @return a string representation of this GrappaSize. 
     */
    public String toFormattedString(String format) {
	return(GrappaSupportPrintf.sprintf(new Object[] { format, this }));
    }

    /**
     * Provides a generic string representation of this object.
     * 
     * @return a generic string representation of this GrappaSize. 
     */
    public String toString() {
	return(width+","+height);
    }
}
