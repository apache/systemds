/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.functionobjects;

import java.io.IOException;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.util.MapReduceTool;


public class RenameFile extends FileFunction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	public static RenameFile singleObj = null;

	private RenameFile() {
		// nothing to do here
	}
	
	public static RenameFile getRenameFileFnObject() {
		if ( singleObj == null )
			singleObj = new RenameFile();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	@Override
	public String execute (String origName, String newName) throws DMLRuntimeException {
		try {
			MapReduceTool.renameFileOnHDFS(origName, newName);
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
		return null;
	}

}
