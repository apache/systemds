/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */


package org.apache.sysml.parser;

import java.util.ArrayList;

public class DMLParseException extends Exception
{
	
	
	/**
	 * The version identifier for this Serializable class.
	 * Increment only if the <i>serialized</i> form of the
	 * class changes.
	 */
	private static final long serialVersionUID = 1L;
	
	private String _filename;
	
	private ArrayList<DMLParseException> _exceptionList;
	
	public ArrayList<DMLParseException> getExceptionList(){
		return _exceptionList;
	}
	
	public DMLParseException(String fname){
		super();
		_filename = fname;
		_exceptionList = new ArrayList<DMLParseException>();
	}
	
	public DMLParseException(String fname, String msg){
		super(msg);
		_filename = fname;
		_exceptionList = new ArrayList<DMLParseException>();
		_exceptionList.add(this);
	}
	
	public DMLParseException(String fname, Exception e){
		super();
		_filename = fname;
		_exceptionList = new ArrayList<DMLParseException>();
		String msg = e.getMessage();
		_exceptionList.add(new DMLParseException(_filename, msg));
	}
	
	public int size(){
		return _exceptionList.size();
	}
	
	public void add(Exception e){
		if (e instanceof DMLParseException)
			_exceptionList.addAll(((DMLParseException)e).getExceptionList());
		else
			_exceptionList.add(new DMLParseException(this._filename, e));
	}
}