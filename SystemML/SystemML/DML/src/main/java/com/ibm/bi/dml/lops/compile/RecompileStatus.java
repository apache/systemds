/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops.compile;

import java.util.HashMap;

import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;

public class RecompileStatus 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private HashMap<String, MatrixCharacteristics> _lastTWrites = null; 
	
	public RecompileStatus()
	{
		_lastTWrites = new HashMap<String,MatrixCharacteristics>();
	}
	
	public HashMap<String, MatrixCharacteristics> getTWriteStats()
	{
		return _lastTWrites;
	}
	
	public void clearStatus()
	{
		_lastTWrites.clear();
	}
}
