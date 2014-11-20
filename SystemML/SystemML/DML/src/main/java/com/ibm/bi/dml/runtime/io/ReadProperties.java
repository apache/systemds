/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.io;

import com.ibm.bi.dml.runtime.matrix.data.FileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;

public class ReadProperties 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	// Properties common to all file formats 
	public String path;
	public long rlen, clen;
	public int brlen, bclen;
	public double expectedSparsity;
	public InputInfo inputInfo;
	public boolean localFS;
	
	// Properties specific to CSV files
	public FileFormatProperties formatProperties;
	
	public ReadProperties() {
		rlen = -1;
		clen = -1;
		brlen = -1;
		bclen = -1;
		expectedSparsity = 0.1d;
		inputInfo = null;
		localFS = false;
	}
}
