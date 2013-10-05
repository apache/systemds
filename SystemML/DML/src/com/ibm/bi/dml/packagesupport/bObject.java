/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.packagesupport;

/**
 * Class to represent an object.
 * 
 */

public class bObject extends FIO 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = 314464073593116450L;
	Object o;

	/**
	 * constructor that takes object as param
	 * 
	 * @param o
	 */
	public bObject(Object o) {
		super(Type.Object);
		this.o = o;
	}

	/**
	 * Method to retrieve object.
	 * 
	 * @return
	 */
	public Object getObject() {
		return o;
	}

}
