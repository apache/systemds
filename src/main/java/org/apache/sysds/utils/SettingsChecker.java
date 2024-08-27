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

package org.apache.sysds.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * Settings Checker class that contains checks for the JVM setting of systemds when executed. These checks helps users
 * configure systemds correctly in case the settings were incorrectly set.
 */
public interface SettingsChecker {
	public static final Log LOG = LogFactory.getLog(SettingsChecker.class.getName());

	/**
	 * Check if the settings set are correct, otherwise write warnings to a user.
	 */
	public static void check() {
		if(LOG.isWarnEnabled()) {
			checkMemorySetting();
		}
	}

	public static void checkMemorySetting() {
		long JRE_Mem_Byte = Runtime.getRuntime().maxMemory();
		long Sys_Mem_Byte = maxMemMachine() * 1024;
		// Default 500MB
		final long DefaultJava_500MB = 1024L * 1024 * 500;
		// 10 GB
		final long Logging_Limit = 1024L * 1024 * 1024 * 10;

		if(JRE_Mem_Byte <= DefaultJava_500MB) {
			String st = byteMemoryToString(JRE_Mem_Byte);
			LOG.warn("Low memory budget set of: " + st + " this should most likely be increased");
		}
		else if(JRE_Mem_Byte < Logging_Limit && JRE_Mem_Byte * 10 < Sys_Mem_Byte) {
			String st = byteMemoryToString(JRE_Mem_Byte);
			String sm = byteMemoryToString(Sys_Mem_Byte);
			LOG.warn("Low memory budget of total: " + sm + " set to: " + st);
		}
	}

	public static long maxMemMachine() {
		String sys = System.getProperty("os.name");
		if("Linux".equals(sys)) {
			return maxMemMachineLinux();
		}
		else if(sys.contains("Mac OS")) {
			return maxMemMachineOSX();
		}
		else if(sys.startsWith("Windows")) {
			return maxMemMachineWin();
		}
		else {
			return -1;
		}
	}

	private static long maxMemMachineLinux() {
		try(BufferedReader reader = new BufferedReader(new FileReader("/proc/meminfo"));) {
			String currentLine = reader.readLine();
			while(!currentLine.contains("MemTotal:"))
				currentLine = reader.readLine();
			return Long.parseLong(currentLine.split(":")[1].split("kB")[0].strip());
		}
		catch(Exception e) {
			e.printStackTrace();
			return -1;
		}
	}

	private static long maxMemMachineOSX() {
		try {
			String command = "sysctl hw.memsize";
			Runtime rt = Runtime.getRuntime();
			Process pr = rt.exec(command);
			String memStr = new String(pr.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
			return Long.parseLong(memStr.trim().substring(12, memStr.length()-1));
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	private static long maxMemMachineWin() {
		try {
			String command = "wmic memorychip get capacity";
			Runtime rt = Runtime.getRuntime();
			Process pr = rt.exec(command);
			String[] memStr = new String(pr.getInputStream().readAllBytes(), StandardCharsets.UTF_8).split("\n");
			//skip header, and aggregate DIMM capacities
			long capacity = 0;
			for( int i=1; i<memStr.length; i++ ) {
				String tmp = memStr[i].trim();
				if( tmp.length() > 0 )
					capacity += Long.parseLong(tmp);
			}
			return capacity;
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * Converts a number of bytes in a long to a human readable string with GB, MB, KB and B.
	 * 
	 * @param bytes Number of bytes.
	 * @return A human readable string
	 */
	public static String byteMemoryToString(long bytes) {
		if(bytes > 1000000000)
			return String.format("%6d GB", bytes / 1024 / 1024 / 1024);
		else if(bytes > 1000000)
			return String.format("%6d MB", bytes / 1024 / 1024);
		else if(bytes > 1000)
			return String.format("%6d KB", bytes / 1024);
		else
			return String.format("%6d B", bytes);
	}
}
