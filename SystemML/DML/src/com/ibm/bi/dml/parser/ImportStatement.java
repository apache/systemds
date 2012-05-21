package com.ibm.bi.dml.parser;

import java.io.File;
import java.util.ArrayList;

import com.ibm.bi.dml.utils.LanguageException;

 
public class ImportStatement extends Statement{
	
	// indicates the imported list of function names 
	private ArrayList<String> _importVarList;
	
	// Indicates if the "import all" option (import moudle (*)) is being used
	private boolean _importAll;
	
	// Stores the alias for the module name
	private String _alias;
	
	// stores only PATH to the module DML file (but NOT the name of the DML file with the module)
	private String _modulePath;
	
	// Stores module name (i.e., the name of the DML file storing the module)
	private String _moduleName;
	
	// Stores complete path to module (package "root" path + modulePath + moduleName + ".dml")
	private String _completePath;
	
	// Stores the absolute path to module
	private String _absolutePath;
	

	public ImportStatement(){
		_importVarList = new ArrayList<String>();
		_importAll = false;
		_alias = null;
		_modulePath = null;
		_moduleName = null;
		_completePath = null;
		_absolutePath = null;
	}
	
	public void setAbsolutePath(String passed){
		_absolutePath = passed;
	}
	
	public void setImportVarList(ArrayList<String> passed){
		_importVarList = passed;
	}
	
	public void addVar(String passed){
		_importVarList.add(passed);
	}
	
	public void setImportAll(boolean passed){
		_importAll = passed;
	}
	
	/**
	 * setModulePathAndName: Set the package (i.e., directory path relative to 
	 * package root directories specified in a dml-path statement.
	 * 
	 * @param pathPieces the individual pieces of the package path, such that when concatenated with 
	 * a "/" will yield the correct entire 
	 */
	public void setModulePathAndName(ArrayList<String> pathPieces ){
		
		_modulePath = new String();
		for (int i=0; i<pathPieces.size()-1; i++)
			_modulePath += pathPieces.get(i) +"/";
		
		// the module name is the name of the dml file 
		_moduleName = pathPieces.get(pathPieces.size()-1);
	}
	
	public void setAlias(String passed){
		_alias = passed;
	}
	
	public String getModuleName() {
		return _moduleName;
	}
	
	public String getModulePath() {
		return _modulePath;
	}
	
	public String getAlias(){
		return _alias;
	}
	
	public boolean getImportAll(){
		return _importAll;
	}
	
	public ArrayList<String> getImportVarList(){
		return _importVarList;
	}

	public String getCompletePath(){
		return _completePath;
	}
	
	public String getAbsolutePath(){
		return _absolutePath;
	}
	
	/**
	 * verify: verifies that the module actually exists and sets the complete 
	 * path for the import statement (package root specified in dml-path concatenated 
	 * with package path 
	 *  
	 * @param packagePaths specifies directories to search for DML packages
	 * 
	 * @return true if the module exists in one of the specified packagePaths
	 */
	public boolean verify(ArrayList<String> packagePaths){
		
		// go through package paths in order
		// find first package path to contain module path
		for (String packagePath : packagePaths){
						
			// translate modulePath to correct path (replace :: with /, 
			// add ".dml" to end to get DML filename containing module)	
			String path = packagePath + "/" + _modulePath + _moduleName + ".dml";
			File completePathFile = new File(path);
			if (completePathFile.exists()){
				_completePath = path;
				_absolutePath = completePathFile.getAbsolutePath();
				return true;
			}	
		} // end for (String packagePath : packagePaths){
		
		_completePath = null;
		_absolutePath = null;
		return false;
	} // end method
	
	public Statement rewriteStatement(String prefix) throws LanguageException{
		throw new LanguageException("rewriting for inlining not supported for ImportStatement");
	}
		
	public void initializeforwardLV(VariableSet activeIn){}
	
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}
	
	@Override
	public VariableSet variablesRead() {
		return null;
	}

	@Override
	public VariableSet variablesUpdated() {
	  	return null;
	}

	@Override
	public boolean controlStatement() {
		return false;
	}

	public String toString(){
		 StringBuffer sb = new StringBuffer();
		 sb.append(Statement.IMPORT + " ");
		 sb.append(this._modulePath + this._moduleName);
		 if (this._importAll){
			sb.append("(*) " ); 
		 }
		 else if (this._importVarList != null && this._importVarList.size() > 0){
			sb.append("(");
			for (int i = 0; i < this._importVarList.size(); i++){
				sb.append(this._importVarList.get(i));
				if (i < this._importVarList.size() -1) sb.append(",");
			}
			sb.append(") ");
		 }
		 if (this._alias != null){
			 sb.append(" AS " + this._alias);
		 }
		 sb.append(";");
		 return sb.toString(); 
	}
	 
}
