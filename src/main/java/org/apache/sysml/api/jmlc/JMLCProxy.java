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

package org.apache.sysml.api.jmlc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

import org.apache.sysml.runtime.instructions.Instruction;

/**
 * This proxy provides thread-local access to output variables per connection
 * in order to enable dynamic recompilation in JMLC.
 */
public class JMLCProxy
{
	private static ThreadLocal<HashSet<String>> _outputs = new ThreadLocal<HashSet<String>>() {
		@Override 
		protected HashSet<String> initialValue() { 
			return null;
		}
	};
	
	public static void setActive(String[] output) {
		if( output != null )
			_outputs.set(new HashSet<String>(Arrays.asList(output)));
		else
			_outputs.remove();
	}

	public static boolean isActive() {
		return (_outputs.get() != null);
	}

	public static ArrayList<Instruction> performCleanupAfterRecompilation(ArrayList<Instruction> tmp) {
		return JMLCUtils.cleanupRuntimeInstructions(tmp, _outputs.get());
	}
}
