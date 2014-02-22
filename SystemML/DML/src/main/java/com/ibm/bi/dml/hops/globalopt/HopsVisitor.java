/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt;

import com.ibm.bi.dml.hops.Hop;

/**
 * An interface for classes that traverse dags of {@link Hops} instances. 
 */
public interface HopsVisitor {

	
	/**
	 * Flags to possibly omit inputs or outputs.
	 */
	public enum Flag {GO_ON, STOP_INPUT, STOP_OUTPUT};
	
	/**
	 * Visit before any edge has been visited.
	 * @param hops
	 * @return
	 */
	public Flag preVisit(Hop hops);
	
	/**
	 * Visit after all incoming edges (according to the visit direction!!!) have been visited.
	 * @param hops
	 * @return
	 */
	public Flag visit(Hop hops);
	
	/**
	 * Visit after every edge has been traversed.
	 * @param hops
	 * @return
	 */
	public Flag postVisit(Hop hops);
	
	/**
	 * Possibility to determine a path in the {@link HopsDag} along which the visitor can tranverse and 
	 * propagate transformations.
	 * @param hops the next visit
	 * @return
	 */
	public boolean matchesPattern(Hop hops);
	
	/**
	 * Determines whether the traversal is done from sink to source or vice versa.
	 * @return true if traversal is required from source to sink.
	 */
	public boolean traverseBackwards();
}
