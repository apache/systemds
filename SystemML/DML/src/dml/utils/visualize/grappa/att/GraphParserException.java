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
 * This class is used whenever a problem is detected during parsing.
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public class GraphParserException extends RuntimeException
{
  /**
   * Constructs an <code>GraphParserException</code> with no detail  message.
   */
  public GraphParserException() {}
  /**
   * Constructs an <code>GraphParserException</code> with the specified 
   * detail message. 
   *
   * @param   message   the detail message.
   */
  public GraphParserException(String message) {
    super(message);
  }
}
