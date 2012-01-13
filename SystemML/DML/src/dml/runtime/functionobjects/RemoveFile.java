package dml.runtime.functionobjects;

import java.io.IOException;

import dml.runtime.util.MapReduceTool;
import dml.utils.DMLRuntimeException;

public class RemoveFile extends FileFunction {
	
	public static RemoveFile singleObj = null;

	private RemoveFile() {
		// nothing to do here
	}
	
	public static RemoveFile getRemoveFileFnObject() {
		if ( singleObj == null )
			singleObj = new RemoveFile();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}

	@Override
	public String execute (String fname) throws DMLRuntimeException {
		try {
			MapReduceTool.deleteFileIfExistOnHDFS(fname);
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
		return null;
	}

}
