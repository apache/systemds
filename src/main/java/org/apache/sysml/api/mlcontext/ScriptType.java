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

package org.apache.sysml.api.mlcontext;

/**
 * ScriptType represents the type of script, DML (R-like syntax) or PYDML
 * (Python-like syntax).
 *
 */
public enum ScriptType {
	/**
	 * R-like syntax.
	 */
	DML,

	/**
	 * Python-like syntax.
	 */
	PYDML;

	/**
	 * Obtain script type as a lowercase string ("dml" or "pydml").
	 * 
	 * @return lowercase string representing the script type
	 */
	public String lowerCase() {
		return super.toString().toLowerCase();
	}

	/**
	 * Is the script type DML?
	 * 
	 * @return {@code true} if the script type is DML, {@code false} otherwise
	 */
	public boolean isDML() {
		return (this == ScriptType.DML);
	}

	/**
	 * Is the script type PYDML?
	 * 
	 * @return {@code true} if the script type is PYDML, {@code false} otherwise
	 */
	public boolean isPYDML() {
		return (this == ScriptType.PYDML);
	}

}
