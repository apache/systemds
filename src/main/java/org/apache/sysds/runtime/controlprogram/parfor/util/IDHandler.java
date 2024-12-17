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

package org.apache.sysds.runtime.controlprogram.parfor.util;

import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.Enumeration;

/**
 * Functionalities for extracting numeric IDs from Hadoop taskIDs and other
 * things related to modification of IDs.
 * 
 * NOTE: Many those functionalities rely on a big endian format. This is always given because 
 * Java stores everything as big endian, independent of the platform. Furthermore, we
 * rely on Long.MAX_VALUE in order to prevent numeric overflows with regard to signed data types.
 * 
 */
public class IDHandler 
{
	public static int extractIntID( String taskID ) {
		int maxlen = (int)(Math.log10(Integer.MAX_VALUE));
		int intVal = (int)extractID( taskID, maxlen );
		return intVal;
	}

	public static long concatIntIDsToLong( int part1, int part2 ) {
		//big-endian version (in java uses only big endian)
		long value = ((long)part1) << 32; //unsigned shift of part1 to first 4bytes
		value = value | part2;            //bitwise OR with part2 (second 4bytes)

		//*-endian version 
		//long value = ((long)part1)*(long)Math.pow(2, 32);
		//value += part2;

		return value;
	}

	/**
	 * Extract int ID from long value
	 * 
	 * @param val long value
	 * @param part if part is 1, use first 4 bytes. if part is 2, use second 4 bytes!
	 * @return return int id, or -1 if part is not 1 or 2!
	 */
	public static int extractIntIDFromLong( long val, int part ) {
		int ret = -1;
		if( part == 1 )
			ret = (int)(val >>> 32);
		else if( part == 2 )
			ret = (int)val; 

		return ret;
	}

	/**
	 * Creates a unique identifier with the pattern &lt;process_id&gt;_&lt;host_ip&gt;.
	 * 
	 * @return distributed unique id
	 */
	public static String createDistributedUniqueID() {
		String uuid = "0_0.0.0.0";

		try {
			long pid = getProcessID();
			String host = getIPAddress(false);
			uuid = pid + "_" + host;
		}
		catch(Exception ex) {}

		return uuid;
	}

	public static long getProcessID() {
		//alternative: ManagementFactory.getRuntimeMXBean().getName() --> pid@hostname
		return ProcessHandle.current().pid();
	}
	
	public static String getIPAddress(boolean noLocal) throws SocketException, UnknownHostException {
		// a host might have many network interfaces, in order to extract the
		// best IP we explicitly filter out 192.168.0.x, 127.0.0.1, and network interfaces
		String ipAddr = null;
		if( noLocal ) {
			Enumeration<NetworkInterface> e = NetworkInterface.getNetworkInterfaces();
			while(e.hasMoreElements()) {
				Enumeration<InetAddress> ee = e.nextElement().getInetAddresses();
				while (ee.hasMoreElements()) {
					String tmp = ee.nextElement().getHostAddress();
					if( tmp!=null && !tmp.contains("192.168.0.") && !tmp.contains("127.0.0.1") && !tmp.contains(":") )
						ipAddr = tmp;
				}
			}
		}
		
		if( ipAddr == null ) {
			//default, which might give local network addresses (e.g., 192.168.0.xxx)
			ipAddr = InetAddress.getLocalHost().getHostAddress();
		}
		return ipAddr;
	}

	private static long extractID( String taskID, int maxlen ) {
		//in: e.g., task_local_0002_m_000009 or task_201203111647_0898_m_000001
		//out: e.g., 2000009

		//generic parsing for flexible taskID formats
		char[] c = taskID.toCharArray(); //all chars
		long value = 0; //1 catch leading zeros as well		
		int count = 0;

		for( int i=c.length-1; i >= 0 && count<maxlen; i-- ) //start at end
		{
			if( c[i] >= 48 && c[i]<=57 )  //'0'-'9'
			{
				long newVal = (c[i]-48);
				newVal = newVal * (long)Math.pow(10, count); //shift to front
				
				value += newVal;
				count++;
			}
		}

		return value;
	}
}
