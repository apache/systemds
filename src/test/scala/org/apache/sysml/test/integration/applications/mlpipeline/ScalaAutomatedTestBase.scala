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
package org.apache.sysml.test.integration.applications.mlpipeline

import java.io.File
import org.apache.sysml.api.ml.ScriptsUtils

object ScalaAutomatedTestBase {
  // *** HACK ALERT *** HACK ALERT *** HACK ALERT ***
	// Hadoop 2.4.1 doesn't work on Windows unless winutils.exe is available 
	// under $HADOOP_HOME/bin and hadoop.dll is available in the Java library
	// path. The following static initializer sets up JVM variables so that 
	// Hadoop can find these native binaries, assuming that any Hadoop code
	// loads after this class and that the JVM's current working directory
	// is the root of this project.	
	val osname = System.getProperty("os.name").toLowerCase();
	if (osname.contains("win")) {
		System.err.printf("AutomatedTestBase has detected a Windows OS and is overriding\n"
				+ "hadoop.home.dir and java.library.path.\n");
		val cwd = System.getProperty("user.dir");

		System.setProperty("hadoop.home.dir", cwd + File.separator
				+ "\\src\\test\\config\\hadoop_bin_windows");
		System.setProperty("java.library.path", cwd + File.separator
				+ "\\src\\test\\config\\hadoop_bin_windows\\bin");
		

	    // Need to muck around with the classloader to get it to use the new
		// value of java.library.path.
		val sysPathsField = classOf[ClassLoader].getDeclaredField("sys_paths");
		sysPathsField.setAccessible(true);
	  sysPathsField.set(null, null);
		// IBM Java throws an exception here, so don't print the stack trace.
	}

	// This ensures that MLPipeline wrappers get appropriate paths to the scripts
	ScriptsUtils.setSystemmlHome(System.getProperty("user.dir") + File.separator + "scripts")
	// *** END HACK ***
}