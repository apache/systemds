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

import java.util.Collections;
import java.util.Map;

import org.apache.sysml.api.mlcontext.ScriptType;
import org.apache.sysml.parser.common.CommonSyntacticValidator;
import org.apache.sysml.parser.dml.DMLParserWrapper;

public class ParserFactory {

	public static ParserWrapper createParser(ScriptType scriptType) {
		return createParser(scriptType, Collections.emptyMap());
	}
	
	/**
	 * Factory method for creating parser wrappers
	 * 
	 * @param scriptType type of script
	 * @param nsscripts map of namespace scripts (name, script)
	 * @return parser wrapper (DMLParserWrapper or PyDMLParserWrapper)
	 */
	public static ParserWrapper createParser(ScriptType scriptType, Map<String, String> nsscripts) {
		ParserWrapper ret = null;
		// create the parser instance
		switch (scriptType) {
			case DML: ret = new DMLParserWrapper(); break;
		}
		CommonSyntacticValidator.init(nsscripts);
		return ret;
	}
}
