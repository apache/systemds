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

package org.apache.sysds.resource;

import org.apache.sysds.resource.enumeration.EnumerationUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

public abstract class CloudUtils {
	public enum CloudProvider {
		AWS // potentially AZURE, GOOGLE
	}
	public enum InstanceType {
		// AWS EC2 instance
		M5, M5A, M6I, M6A, M6G, M7I, M7A, M7G, // general purpose - vCores:mem~=1:4
		C5, C5A, C6I, C6A, C6G, C7I, C7A, C7G, // compute optimized - vCores:mem~=1:2
		R5, R5A, R6I, R6A, R6G, R7I, R7A, R7G; // memory optimized - vCores:mem~=1:8
		// Potentially VM instance types for different Cloud providers

		public static InstanceType customValueOf(String name) {
			return InstanceType.valueOf(name.toUpperCase());
		}
	}

	public enum InstanceSize {
		_XLARGE, _2XLARGE, _4XLARGE, _8XLARGE, _12XLARGE, _16XLARGE, _24XLARGE, _32XLARGE, _48XLARGE;
		// Potentially VM instance sizes for different Cloud providers

		public static InstanceSize customValueOf(String name) {
			return InstanceSize.valueOf("_"+name.toUpperCase());
		}
	}

	public static final String SPARK_VERSION = "3.3.0";
	public static final double MINIMAL_EXECUTION_TIME = 120; // seconds; NOTE: set always equal or higher than DEFAULT_CLUSTER_LAUNCH_TIME
	public static final double DEFAULT_CLUSTER_LAUNCH_TIME = 120; // seconds; NOTE: set always to at least 60 seconds

	public static long GBtoBytes(double gb) {
		return (long) (gb * 1024 * 1024 * 1024);
	}
	public abstract boolean validateInstanceName(String instanceName);
	public abstract InstanceType getInstanceType(String instanceName);
	public abstract InstanceSize getInstanceSize(String instanceName);

	/**
	 * This method calculates the cluster price based on the
	 * estimated execution time and the cluster configuration.
	 * @param config the cluster configuration for the calculation
	 * @param time estimated execution time in seconds
	 * @return price for the given time
	 */
	public abstract double calculateClusterPrice(EnumerationUtils.ConfigurationPoint config, double time);

	/**
	 * Performs read of csv file filled with VM instance characteristics.
	 * Each record in the csv should carry the following information (including header):
	 * <ul>
	 * <li>API_Name - naming for VM instance used by the provider</li>
	 * <li>Memory - floating number for the instance memory in GBs</li>
	 * <li>vCPUs - number of physical threads</li>
	 * <li>gFlops - FLOPS capability of the CPU in GFLOPS (Giga)</li>
	 * <li>ramSpeed - memory bandwidth in MB/s</li>
	 * <li>diskSpeed - memory bandwidth in MB/s</li>
	 * <li>networkSpeed - memory bandwidth in MB/s</li>
	 * <li>Price - price for instance per hour</li>
	 * </ul>
	 * @param instanceTablePath csv file
	 * @return map with filtered instances
	 * @throws IOException in case problem at reading the csv file
	 */
	public HashMap<String, CloudInstance> loadInstanceInfoTable(String instanceTablePath) throws IOException {
		HashMap<String, CloudInstance> result = new HashMap<>();
		int lineCount = 1;
		// try to open the file
		try(BufferedReader br = new BufferedReader(new FileReader(instanceTablePath))){
			String parsedLine;
			// validate the file header
			parsedLine = br.readLine();
			if (!parsedLine.equals("API_Name,Memory,vCPUs,gFlops,ramSpeed,diskSpeed,networkSpeed,Price"))
				throw new IOException("Invalid CSV header inside: " + instanceTablePath);
	
	
			while ((parsedLine = br.readLine()) != null) {
				String[] values = parsedLine.split(",");
				if (values.length != 8 || !validateInstanceName(values[0]))
					throw new IOException(String.format("Invalid CSV line(%d) inside: %s", lineCount, instanceTablePath));
	
				String API_Name = values[0];
				long Memory = (long) (Double.parseDouble(values[1])*1024)*1024*1024;
				int vCPUs = Integer.parseInt(values[2]);
				double gFlops = Double.parseDouble(values[3]);
				double ramSpeed = Double.parseDouble(values[4]);
				double diskSpeed = Double.parseDouble(values[5]);
				double networkSpeed = Double.parseDouble(values[6]);
				double Price = Double.parseDouble(values[7]);
	
				CloudInstance parsedInstance = new CloudInstance(
						API_Name,
						Memory,
						vCPUs,
						gFlops,
						ramSpeed,
						diskSpeed,
						networkSpeed,
						Price
				);
				result.put(API_Name, parsedInstance);
				lineCount++;
			}
		}

		return result;
	}
}
