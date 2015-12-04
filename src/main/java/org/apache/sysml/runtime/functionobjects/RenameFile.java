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

package org.apache.sysml.runtime.functionobjects;

import java.io.IOException;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.util.MapReduceTool;


public class RenameFile extends FileFunction 
{
		
	private static RenameFile singleObj = null;

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
