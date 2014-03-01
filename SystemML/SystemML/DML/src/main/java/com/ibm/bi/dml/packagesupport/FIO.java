/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.packagesupport;

import java.io.Serializable;

/**
 * abstract class to represent all input and output objects for package
 * functions.
 * 
 * 
 * 
 */

public abstract class FIO implements Serializable
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	

	private static final long serialVersionUID = 1189133371204708466L;
	Type type;

	/**
	 * Constructor to set type
	 * 
	 * @param type
	 */
	public FIO(Type type) {
		this.type = type;
	}

	/**
	 * Method to get type
	 * 
	 * @return
	 */
	public Type getType() {
		return type;
	}

}
