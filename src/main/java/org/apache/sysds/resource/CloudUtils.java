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
import org.apache.wink.json4j.JSONObject;
import org.apache.wink.json4j.JSONArray;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;

/**
 * Class providing static utilities for cloud related operations.
 * Some of the utilities are dependent on the cloud provider,
 * but currently only AWS is supported and set as default provider.
 * The method {@code setProvider()} is to be used for setting terget
 * provider once more providers become supported.
 */
public class CloudUtils {
	public enum CloudProvider {
		AWS // potentially AZURE, GOOGLE
	}
	public enum InstanceFamily {
		// AWS EC2 instance
		M5, M5A, M6I, M6A, M6G, M7I, M7A, M7G, // general purpose - vCores:mem~=1:4
		M5D, M5AD, M6ID, M6GD, M7GD, // with attached Instance Store (NVMe-based SSD)
		M5N, M6IN, M5DN, M6IDN, // with enhanced network capabilities (and Instance Store)
		M5ZN, // with more powerful CPU and enhanced networking
		C5, C5A, C6I, C6A, C6G, C7I, C7A, C7G, // compute optimized - vCores:mem~=1:2
		C5D, C5AD, C6ID, C6GD, C7GD, // with attached Instance Store
		C5N, C6IN, C6GN, C7GN, // with enhanced network capabilities
		R5, R5A, R6I, R6A, R6G, R7I, R7A, R7G, // memory optimized - vCores:mem~=1:8
		R5D, R5AD, R6ID, R6AD, R6GD, R7ID, R7AD, R7GD, // with attached Instance Store
		R5N, R5DN, R6IN, R6IDN // with attached Instance Store
		;// Potentially VM instance families for different Cloud providers
		public static InstanceFamily customValueOf(String name) {
			return InstanceFamily.valueOf(name.toUpperCase());
		}
	}

	public enum InstanceSize {
		_XLARGE, _2XLARGE, _3XLARGE, _4XLARGE, _6XLARGE, _8XLARGE, _9XLARGE, _12XLARGE, _16XLARGE, _18XLARGE, _24XLARGE, _32XLARGE, _48XLARGE;
		// Potentially VM instance sizes for different Cloud providers

		public static InstanceSize customValueOf(String name) throws IllegalArgumentException {
			return InstanceSize.valueOf("_"+name.toUpperCase());
		}
	}

	public static final String EC2_REGEX = "^([a-z]+)([0-9])(a|g|i?)([bdnez]*)\\.([a-z0-9]+)$";
	public static final String SPARK_VERSION = "3.3.0";
	public static final double MINIMAL_EXECUTION_TIME = 600; // seconds; NOTE: set always equal or higher than DEFAULT_CLUSTER_LAUNCH_TIME
	public static final double DEFAULT_CLUSTER_LAUNCH_TIME = 600; // seconds; NOTE: set always to at least 60 seconds

	/**
	 * Flag to mark the target provider for the utilities.
	 */
	private static CloudProvider provider = CloudProvider.AWS;

	/**
	 * Static prover initialization method.
	 *
	 * @param provider target provider
	 */
	public static void setProvider(CloudProvider provider) {
		CloudUtils.provider = provider;
	}
	private CloudUtils() {} // only static methods does not require class initialization

	public static long GBtoBytes(double gb) {
		return (long) (gb * 1024 * 1024 * 1024);
	}
	public static boolean validateInstanceName(String instanceName) {
		instanceName = instanceName.toLowerCase();
		if (provider == CloudProvider.AWS && !instanceName.toLowerCase().matches(EC2_REGEX)) return false;
		try {
			getInstanceFamily(instanceName);
			getInstanceSize(instanceName);
		} catch (IllegalArgumentException e) {
			return false;
		}
		return true;
	}
	public static InstanceFamily getInstanceFamily(String instanceName) {
		String familyAsString = instanceName.split("\\.")[0];
		// throws exception if string value is not valid
		return InstanceFamily.customValueOf(familyAsString);
	}
	public static InstanceSize getInstanceSize(String instanceName) {
		String sizeAsString = instanceName.split("\\.")[1];
		// throws exception if string value is not valid
		return InstanceSize.customValueOf(sizeAsString);
	}


	/**
	 * This method calculates the cluster price based on the
	 * estimated execution time and the cluster configuration.
	 * The calculation considers extra storage price for Spark cluster
	 * because of the HDFS dependency, but the costs for the root storage
	 * is not accounted for here.
	 *
	 * @param config   the cluster configuration for the calculation
	 * @param time     estimated execution time in seconds
	 * @param provider cloud provider for the instances of the cluster
	 * @return price for the given time
	 */
	public static double calculateClusterPrice(EnumerationUtils.ConfigurationPoint config, double time, CloudProvider provider) {
		double pricePerSecond;
		if (provider == CloudProvider.AWS) {
			if (config.numberExecutors == 0) { // single instance (no cluster management -> no extra fee)
				// price = EC2 price, no extra storage needed (only root volume, which is not accounted for)
				CloudInstance singleNode = config.driverInstance;
				pricePerSecond = singleNode.getPrice();
			} else {
				// price = EC2 price + EMR fee + extra storage (EBS) price
				CloudInstance masterNode = config.driverInstance;
				CloudInstance coreNode = config.executorInstance;
				pricePerSecond = masterNode.getPrice() + masterNode.getExtraFee() + masterNode.getExtraStoragePrice();
				pricePerSecond += config.numberExecutors * (coreNode.getPrice() + coreNode.getExtraFee() + coreNode.getExtraStoragePrice());
			}
		} else {
			throw new IllegalArgumentException("AWS is the only cloud provider supported at the moment");
		}
		return time * pricePerSecond;
	}

	/**
	 * Performs read of csv file filled with relevant AWS fees/prices per region.
	 * Each record in the csv should carry the following information (including header):
	 * <ul>
	 * <li>Region - AWS region abbreviation</li>
	 * <li>Fee Ratio - Ratio of EMR fee per instance to EC2 price per instance per hour</li>
	 * <li>EBS Price- Price for EBS per month per GB</li>
	 * </ul>
	 * @param feeTablePath csv file path
	 * @param region AWS region abbreviation
	 * @return static array of doubles with 2 elements: [EMR fee ratio, EBS price]
	 * @throws IOException in case of invalid file format
	 */
	public static double[] loadRegionalPrices(String feeTablePath, String region) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(feeTablePath));
		String parsedLine;
		// validate the file header
		parsedLine = br.readLine();
		if (!parsedLine.equals("Region,Fee Ratio,EBS Price"))
			throw new IOException("Fee Table: invalid CSV header: " + parsedLine);
		while ((parsedLine = br.readLine()) != null) {
			String[] values = parsedLine.split(",");
			if (values.length != 3)
				throw new IOException(String.format("Fee Table: invalid CSV line '%s' inside: %s", parsedLine, feeTablePath));
			if (region.equals(values[0])) {
				return new double[] { Double.parseDouble(values[1]), Double.parseDouble(values[2]) };
			}
		}
		throw new IOException(String.format("Fee Table: region '%s' not found in the CSV table: %s", region, feeTablePath));
	}

		/**
	 * Performs read of csv file filled with VM instance characteristics.
	 * Each record in the csv should carry the following information (including header):
	 * <ul>
	 * <li>API_Name - naming for VM instance used by the provider</li>
	 * <li>Price - price for instance per hour</li>
	 * <li>Memory - floating number for the instance memory in GBs</li>
	 * <li>vCPUs - number of physical threads</li>
	 * <li>Cores - number of physical cores (not relevant at the moment)</li>
	 * <li>gFlops - FLOPS capability of the CPU in GFLOPS (Giga)</li>
	 * <li>memoryBandwidth - memory bandwidth in MB/s</li>
	 * <li>NVMe - flag if NVMe storage volume(s) are attached</li>
	 * <li>storageVolumes - number of NVMe or EBS (to additionally configured) volumes</li>
	 * <li>sizeVolumes - size of each NVMe or EBS (to additionally configured) volume</li>
	 * <li>diskReadBandwidth - disk read bandwidth in MB/s</li>
	 * <li>diskReadBandwidth - disk write bandwidth in MB/s</li>
	 * <li>networkBandwidth - network bandwidth in MB/s</li>
	 * </ul>
	 * @param instanceTablePath csv file path
	 * @return map with filtered instances
	 * @throws IOException in case problem at reading the csv file
	 */
	public static HashMap<String, CloudInstance> loadInstanceInfoTable(String instanceTablePath, double feeRatio, double storagePrice) throws IOException {
		HashMap<String, CloudInstance> result = new HashMap<>();
		int lineCount = 1;
		// try to open the file
		BufferedReader br = new BufferedReader(new FileReader(instanceTablePath));
		String parsedLine;
		// validate the file header
		parsedLine = br.readLine();
		if (!parsedLine.equals("API_Name,Price,Memory,vCPUs,Cores,GFLOPS,memBandwidth,NVMe,storageVolumes,sizePerVolume,readStorageBandwidth,writeStorageBandwidth,networkBandwidth"))
			throw new IOException("Instance info table: invalid CSV header inside: " + instanceTablePath);


		while ((parsedLine = br.readLine()) != null) {
			String[] values = parsedLine.split(",");
			if (values.length != 13 || !validateInstanceName(values[0]))
				throw new IOException(String.format("Instance info table: invalid CSV line(%d) inside: %s, instance of type %s", lineCount, instanceTablePath, values[0]));

			String name = values[0];
			double price = Double.parseDouble(values[1]);
			double extraFee = price * feeRatio;
			long memory = GBtoBytes(Double.parseDouble(values[2]));
			int vCPUs = Integer.parseInt(values[3]);
			double GFlops = Double.parseDouble(values[5]);
			double memBandwidth = Double.parseDouble(values[6]);
			double diskReadBandwidth = Double.parseDouble(values[10]);
			double diskWriteBandwidth = Double.parseDouble(values[11]);
			double networkBandwidth = Double.parseDouble(values[12]);
			boolean NVMeStorage = Boolean.parseBoolean(values[7]);
			int numberStorageVolumes = Integer.parseInt(values[8]);
			double sizeStorageVolumes = Double.parseDouble(values[9]);

			CloudInstance parsedInstance = new CloudInstance(
					name,
					price,
					extraFee,
					storagePrice,
					memory,
					vCPUs,
					GFlops,
					memBandwidth,
					diskReadBandwidth,
					diskWriteBandwidth,
					networkBandwidth,
					NVMeStorage,
					numberStorageVolumes,
					sizeStorageVolumes
			);
			result.put(name, parsedInstance);
			lineCount++;
		}

		return result;
	}

	/**
	 * Generates json file storing the instance type for
	 * single node executions
	 * @param instance EC2 instance object (always set one)
	 * @param filePath path for the json file
	 */
	public static void generateEC2ConfigJson(CloudInstance instance, String filePath) {
		try {
			JSONObject ec2Config = new JSONObject();

			ec2Config.put("InstanceType", instance.getInstanceName());
			ec2Config.put("MinCount", 1);
			ec2Config.put("MaxCount", 1);

			try (FileWriter file = new FileWriter(filePath)) {
				file.write(ec2Config.write(true));
				System.out.println("EC2 configuration JSON file: " + filePath);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Generates json file with instance groups argument for
	 * launching AWS EMR cluster
	 *
	 * @param masterInstance EC2 instance object (always set one)
	 * @param coreInstanceCount number core instances
	 * @param coreInstance EC2 instance object
	 * @param filePath path for the json file
	 */
	public static void generateEMRInstanceGroupsJson(CloudInstance masterInstance, int coreInstanceCount, CloudInstance coreInstance,
												  String filePath) {
		try {
			JSONArray instanceGroups = new JSONArray();

			// Master (Primary) instance group
			JSONObject masterGroup = new JSONObject();
			masterGroup.put("InstanceCount", 1);
			masterGroup.put("InstanceGroupType", "MASTER");
			masterGroup.put("InstanceType", masterInstance.getInstanceName());
			masterGroup.put("Name", "Master Instance Group");
			attachEBSConfigsIfNeeded(masterInstance, masterGroup);
			instanceGroups.add(masterGroup);

			// Core instance group
			JSONObject coreGroup = new JSONObject();
			coreGroup.put("InstanceCount", coreInstanceCount);
			coreGroup.put("InstanceGroupType", "CORE");
			coreGroup.put("InstanceType", coreInstance.getInstanceName());
			coreGroup.put("Name", "Core Instance Group");
			attachEBSConfigsIfNeeded(coreInstance, coreGroup);
			instanceGroups.add(coreGroup);

			try (FileWriter file = new FileWriter(filePath)) {
				file.write(instanceGroups.write(true));
				System.out.println("Instance Groups JSON file created: " + filePath);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void attachEBSConfigsIfNeeded(CloudInstance instance, JSONObject instanceGroup) {
		if (!instance.isNVMeStorage()) {
			try {
				JSONObject volumeSpecification = new JSONObject();
				volumeSpecification.put("SizeInGB", instance.getSizeStorageVolumes());
				volumeSpecification.put("VolumeType", "gp3");

				JSONObject ebsBlockDeviceConfig = new JSONObject();
				ebsBlockDeviceConfig.put("VolumesPerInstance", instance.getNumberStorageVolumes());
				ebsBlockDeviceConfig.put("VolumeSpecification", volumeSpecification);
				JSONArray ebsBlockDeviceConfigsArray = new JSONArray();
				ebsBlockDeviceConfigsArray.add(ebsBlockDeviceConfig);

				JSONObject ebsConfiguration = new JSONObject();
				ebsConfiguration.put("EbsOptimized", true);
				ebsConfiguration.put("EbsBlockDeviceConfigs", ebsBlockDeviceConfigsArray);
				instanceGroup.put("EbsConfiguration", ebsConfiguration);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	/**
	 * Generate json file with configurations attribute for
	 * launching AWS EMR cluster with Spark
	 *
	 * @param filePath path for the json file
	 */
	public static void generateEMRConfigurationsJson(String filePath) {
		try {
			JSONArray configurations = new JSONArray();

			// Spark Configuration
			JSONObject sparkConfig = new JSONObject();
			sparkConfig.put("Classification", "spark");
			sparkConfig.put("Properties", new JSONObject().put("maximizeResourceAllocation", "true"));

			// Spark-env and export JAVA_HOME Configuration
			JSONObject sparkEnvConfig = new JSONObject();
			sparkEnvConfig.put("Classification", "spark-env");

			JSONObject jvmVersion = new JSONObject().put("JAVA_HOME", "/usr/lib/jvm/jre-11");
			JSONObject exportConfig = new JSONObject();
			exportConfig.put("Classification", "export");
			exportConfig.put("Properties", jvmVersion);
			JSONArray jvmArray = new JSONArray();
			jvmArray.add(exportConfig);
			sparkEnvConfig.put("Configurations", jvmArray);
			configurations.add(sparkConfig);
			configurations.add(sparkEnvConfig);

			try (FileWriter file = new FileWriter(filePath)) {
				file.write(configurations.write(true));
				System.out.println("Configurations JSON file created: " + filePath);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
