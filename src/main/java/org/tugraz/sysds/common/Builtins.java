/*
 * Copyright 2018 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.common;

import java.util.EnumSet;
import java.util.HashMap;

/**
 * Enum to represent all builtin functions in the default name space.
 * Each function is either native or implemented by a DML script. In
 * case of DML script, these functions are loaded during parsing. As
 * always, user-defined DML-bodied functions take precedence over all
 * builtin functions.
 * 
 * To add a new builtin script function, simply add the definition here
 * as well as a dml file in script/builtin with a matching name.
 */
public enum Builtins {
	SIGMOD("sigmoid", true);   // 1 / (1 + exp(-X))
	
	
	Builtins(String name, boolean script) {
		_name = name;
		_script = script;
	}
	
	private final static HashMap<String, Builtins> _map = new HashMap<>();
	
	static {
		//materialize lookup map for all builtin names
		for( Builtins b : EnumSet.allOf(Builtins.class) )
			_map.put(b.getName(), b);
	}
	
	private final String _name;
	private final boolean _script;
	
	public String getName() {
		return _name;
	}
	
	public boolean isScript() {
		return _script;
	}
	
	public static boolean contains(String name, boolean scriptOnly) {
		Builtins tmp = _map.get(name);
		return tmp != null 
			&& (!scriptOnly || tmp._script);
	}
}
