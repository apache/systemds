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
 * An extension of the Enumeration interface specific to enumerations of
 * graph elements.
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public interface GraphEnumeration extends java.util.Enumeration
{
  /**
   * Get the root of this enumeration.
   *
   * @return the root subgraph for this enumeration
   */
  public Subgraph getSubgraphRoot();

  /**
   * Get the types of elements possibly contained in this enumeration.
   *
   * @return an indication of the types of elements in this enumeration
   * @see GrappaConstants#NODE
   * @see GrappaConstants#EDGE
   * @see GrappaConstants#SUBGRAPH
   */
  public int getEnumerationTypes();

  /**
   * A convenience method that should just return a cast
   * of a call to nextElement()
   *
   * @return the next graph element in the enumeration
   * @exception java.util.NoSuchElementException whenever the enumeration has no more
   *                                   elements.
   */
  public Element nextGraphElement() throws java.util.NoSuchElementException;
}
