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

package org.tugraz.sysds.utils;

import org.tugraz.sysds.api.mlcontext.MLContext;
import org.tugraz.sysds.api.mlcontext.MLContextException;
import org.tugraz.sysds.parser.Expression;

/**
 * The purpose of this proxy is to shield systemds internals from direct access to MLContext
 * which would try to load spark libraries and hence fail if these are not available. This
 * indirection is much more efficient than catching NoClassDefFoundErrors for every access
 * to MLContext (e.g., on each recompile).
 *
 */
public class MLContextProxy
{

	private static boolean _active = false;

	public static void setActive(boolean flag) {
		_active = flag;
	}

	public static boolean isActive() {
		return _active;
	}
	
	public static void setAppropriateVarsForRead(Expression source, String targetname) {
		MLContext.getActiveMLContext().getInternalProxy().setAppropriateVarsForRead(source, targetname);
	}

	public static MLContext getActiveMLContext() {
		return MLContext.getActiveMLContext();
	}

	public static MLContext getActiveMLContextForAPI() {
		if (MLContext.getActiveMLContext() != null) {
			return MLContext.getActiveMLContext();
		}
		throw new MLContextException("No MLContext object is currently active. Have you created one? "
				+ "Hint: in Scala, 'val ml = new MLContext(sc)'", true);
	}
}
