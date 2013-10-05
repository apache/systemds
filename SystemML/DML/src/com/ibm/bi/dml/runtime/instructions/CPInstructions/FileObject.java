/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


public class FileObject extends Data 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public String toString() {
		return "(" + _name + "," + _filePath + ")" ;
	}
	
	private String _name;
	private String _filePath;
	private boolean _isLocal;
	
	public FileObject(String path){
		this(null,path);
	}

	public FileObject(String name, String path){
		super(DataType.SCALAR, ValueType.STRING);
 		_name = name;
		_filePath = path;
		_isLocal = false;
	}
 
	public void setLocal(){
		_isLocal = true;
	}
	
	public String getName(){
		return _name;
	}
	
	public String getFilePath(){
		return _filePath;
	}

	public boolean isLocal(){
		return _isLocal;
	}

	@Override
	public String getDebugName() {
		// TODO Auto-generated method stub
		return null;
	}
}
