package dml.runtime.instructions.CPInstructions;


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
}
