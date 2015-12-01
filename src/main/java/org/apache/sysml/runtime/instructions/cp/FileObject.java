/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.runtime.instructions.cp;

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;


public class FileObject extends Data 
{
	
	private static final long serialVersionUID = 2057548889127080668L;
	
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
