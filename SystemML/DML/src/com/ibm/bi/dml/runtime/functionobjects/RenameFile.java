package com.ibm.bi.dml.runtime.functionobjects;

import java.io.IOException;

import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class RenameFile extends FileFunction {
	
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
