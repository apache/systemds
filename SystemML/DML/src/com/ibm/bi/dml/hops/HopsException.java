/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.utils.DMLException;

/**
 * <p>Exception occurring in the HOP level.</p>
 */
@SuppressWarnings("serial")
public class HopsException extends DMLException 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
    /**
     * @see java.lang.Exception#Exception()
     */
    public HopsException() {
        super();
    }
    
    /**
     * @see java.lang.Exception#Exception(String)
     */
    public HopsException(String message) {
        super(message);
    }
    
    /**
     * @see java.lang.Exception#Exception(Throwable)
     */
    public HopsException(Throwable cause) {
        super(cause);
    }
    
    /**
     * @see java.lang.Exception#Exception(String, Throwable)
     */
    public HopsException(String message, Throwable cause) {
        super(message, cause);
    }

}
