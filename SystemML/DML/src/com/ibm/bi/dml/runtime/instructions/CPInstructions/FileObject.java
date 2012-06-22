package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


public class FileObject extends Data {

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
