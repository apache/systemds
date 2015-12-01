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

package com.ibm.bi.dml.parser.antlr4;

import java.util.HashMap;
import com.ibm.bi.dml.parser.DMLProgram;

public class StatementInfo {

	public com.ibm.bi.dml.parser.Statement stmt = null;
	
	// Valid only for import statements
	public HashMap<String,DMLProgram> namespaces = null;
	
	// Valid only for function statement
	//public String namespace = DMLProgram.DEFAULT_NAMESPACE;
	public String functionName = "";

}
