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

import org.apache.sysds.resource.enumeration.EnumerationUtils.ConfigurationPoint;
import org.apache.wink.json4j.JSONObject;
import org.apache.wink.json4j.JSONArray;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.rmi.RemoteException;
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
		_XLARGE, _2XLARGE, _3XLARGE, _4XLARGE, _6XLARGE, _8XLARGE, _9XLARGE,
		_12XLARGE, _16XLARGE, _18XLARGE, _24XLARGE, _32XLARGE, _48XLARGE;
		// Potentially VM instance sizes for different Cloud providers

		public static InstanceSize customValueOf(String name) throws IllegalArgumentException {
			return InstanceSize.valueOf("_"+name.toUpperCase());
		}
	}
	public static final double JVM_MEMORY_FACTOR = 0.9; // (10% for the OS and other external processes)

	public static final String EC2_REGEX = "^([a-z]+)([0-9])(a|g|i?)([bdnez]*)\\.([a-z0-9]+)$";
	public static final int EBS_DEFAULT_ROOT_SIZE_EMR = 15; // GB
	public static final int EBS_DEFAULT_ROOT_SIZE_EC2 = 8; // GB
	public static final String SPARK_VERSION = "3.3.0";
	// set always equal or higher than DEFAULT_CLUSTER_LAUNCH_TIME
	public static final double MINIMAL_EXECUTION_TIME = 300; // seconds;
	// set always to a positive value
	public static final double DEFAULT_CLUSTER_LAUNCH_TIME = 300; // seconds;

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
	public static double calculateClusterPrice(ConfigurationPoint config, double time, CloudProvider provider) {
		double pricePerSecond;
		if (provider == CloudProvider.AWS) {
			if (config.numberExecutors == 0) { // single instance (no cluster management -> no extra fee)
				// price = EC2 price + storage price (EBS) - use only the half of the price since
				// the half of the storage will be automatically configured
				// because in single-node mode SystemDS does not utilize HDFS
				// (only minimal root EBS when Instance Store available)
				CloudInstance singleNode = config.driverInstance;
				pricePerSecond = singleNode.getPrice() + (singleNode.getExtraStoragePrice() / 2);
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
		try(BufferedReader br = new BufferedReader(new FileReader(feeTablePath))) {
			// validate the file header
			String parsedLine = br.readLine();
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
		} catch (FileNotFoundException e) {
			throw new RemoteException(feeTablePath+" read failed: "+e);
		}
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
	 *
	 * @param instanceTablePath csv file path
	 * @param emrFeeRatio EMR fee as fraction of the instance price (depends on the region)
	 * @param ebsStoragePrice EBS price per GB per month (depends on the region)
	 * @return map with filtered instances
	 * @throws IOException in case problem at reading the csv file
	 */
	public static HashMap<String, CloudInstance> loadInstanceInfoTable(
			String instanceTablePath, double emrFeeRatio, double ebsStoragePrice) throws IOException {
		// store as mapping the instance type name to the instance object
		HashMap<String, CloudInstance> result = new HashMap<>();
		int lineCount = 1;
		// try to open the file
		try(BufferedReader br = new BufferedReader(new FileReader(instanceTablePath))) {
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
				double extraFee = price * emrFeeRatio;
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
					name, price, extraFee, ebsStoragePrice,
					memory, vCPUs, GFlops, memBandwidth,
					diskReadBandwidth, diskWriteBandwidth,
					networkBandwidth, NVMeStorage,
					numberStorageVolumes, sizeStorageVolumes
				);
				result.put(name, parsedInstance);
				lineCount++;
			}
	
			return result;
		}
		catch(Exception ex) {
			throw new IOException("Read failed", ex);
		}
	}

	/**
	 * Generates json file storing the instance type and relevant characteristics
	 * for single node executions.
	 * The resulting file is to be used only for parsing the attributes and
	 * is not suitable for direct options input to AWS CLI.
	 *
	 * @param instance EC2 instance object (always set one)
	 * @param filePath path for the json file
	 */
	public static void generateEC2ConfigJson(CloudInstance instance, String filePath) {
		try {
			JSONObject ec2Config = new JSONObject();

			ec2Config.put("InstanceType", instance.getInstanceName());
			// EBS size of the root volume (only one volume in this case)
			int ebsRootSize = EBS_DEFAULT_ROOT_SIZE_EC2;
			if (!instance.isNVMeStorage()) // plan for only half of the EMR storage budget
				ebsRootSize += (int) Math.ceil(instance.getNumStorageVolumes()*instance.getSizeStoragePerVolume()/2);
			ec2Config.put("VolumeSize", ebsRootSize);
			ec2Config.put("VolumeType", "gp3");
			ec2Config.put("EbsOptimized", true);
			// JVM memory budget used at resource optimization
			int cpMemory = (int) (instance.getMemory()/ (1024*1024) * JVM_MEMORY_FACTOR);
			ec2Config.put("JvmMaxMemory", cpMemory);

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
	 * @param clusterConfig object representing EMR cluster configurations
	 * @param filePath path for the output json file
	 */
	public static void generateEMRInstanceGroupsJson(ConfigurationPoint clusterConfig, String filePath) {
		try {
			JSONArray instanceGroups = new JSONArray();

			// Master (Primary) instance group
			JSONObject masterGroup = new JSONObject();
			masterGroup.put("InstanceCount", 1);
			masterGroup.put("InstanceGroupType", "MASTER");
			masterGroup.put("InstanceType",clusterConfig.driverInstance.getInstanceName());
			masterGroup.put("Name", "Master Instance Group");
			attachEBSConfigsIfNeeded(clusterConfig.driverInstance, masterGroup);
			instanceGroups.add(masterGroup);

			// Core instance group
			JSONObject coreGroup = new JSONObject();
			coreGroup.put("InstanceCount", clusterConfig.numberExecutors);
			coreGroup.put("InstanceGroupType", "CORE");
			coreGroup.put("InstanceType", clusterConfig.executorInstance.getInstanceName());
			coreGroup.put("Name", "Core Instance Group");
			attachEBSConfigsIfNeeded(clusterConfig.executorInstance, coreGroup);
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
		// in AWS CLI the root EBS volume is configured with a separate optional flag (default 15GB)
		if (!instance.isNVMeStorage()) {
			try {
				JSONObject volumeSpecification = new JSONObject();
				volumeSpecification.put("SizeInGB", (int) instance.getSizeStoragePerVolume());
				volumeSpecification.put("VolumeType", "gp3");

				JSONObject ebsBlockDeviceConfig = new JSONObject();
				ebsBlockDeviceConfig.put("VolumesPerInstance", instance.getNumStorageVolumes());
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
	 * @param clusterConfig object representing EMR cluster configurations
	 * @param filePath path for the output json file
	 */
	public static void generateEMRConfigurationsJson(ConfigurationPoint clusterConfig, String filePath) {
		try {
			JSONArray configurations = new JSONArray();

			// Spark Configuration
			JSONObject sparkConfig = new JSONObject();
			sparkConfig.put("Classification", "spark");
			// do not use the automatic EMR cluster configurations
			sparkConfig.put("Properties", new JSONObject().put("maximizeResourceAllocation", "false"));

			// set custom defined cluster configurations
			JSONObject sparkDefaultsConfig = new JSONObject();
			sparkDefaultsConfig.put("Classification", "spark-defaults");

			JSONObject sparkDefaultsProperties = new JSONObject();
			long driverMemoryBytes = calculateEffectiveDriverMemoryBudget(clusterConfig.driverInstance.getMemory(),
					clusterConfig.numberExecutors * clusterConfig.executorInstance.getVCPUs());
			int driverMemory = (int) (driverMemoryBytes / (1024*1024));
			sparkDefaultsProperties.put("spark.driver.memory", (driverMemory)+"m");
			sparkDefaultsProperties.put("spark.driver.maxResultSize", String.valueOf(0));
			// calculate the exact resource limits for YARN containers to maximize the utilization
			int[] executorResources = getEffectiveExecutorResources(
					clusterConfig.executorInstance.getMemory(),
					clusterConfig.executorInstance.getVCPUs(),
					clusterConfig.numberExecutors
			);
			sparkDefaultsProperties.put("spark.executor.memory", (executorResources[0])+"m");
			sparkDefaultsProperties.put("spark.executor.cores", Integer.toString(executorResources[1]));
			sparkDefaultsProperties.put("spark.executor.instances", Integer.toString(executorResources[2]));
			// values copied from SparkClusterConfig.analyzeSparkConfiguation
			sparkDefaultsProperties.put("spark.storage.memoryFraction", String.valueOf(0.6));
			sparkDefaultsProperties.put("spark.memory.storageFraction", String.valueOf(0.5));
			sparkDefaultsProperties.put("spark.executor.memoryOverheadFactor", String.valueOf(0.1));
			// set the custom AM configurations
			sparkDefaultsProperties.put("spark.yarn.am.memory", (executorResources[3])+"m");
			sparkDefaultsProperties.put("spark.yarn.am.cores", Integer.toString(executorResources[4]));
			sparkDefaultsConfig.put("Properties", sparkDefaultsProperties);

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
			configurations.add(sparkDefaultsConfig);
			configurations.add(sparkEnvConfig);

			try (FileWriter file = new FileWriter(filePath)) {
				file.write(configurations.write(true));
				System.out.println("Configurations JSON file created: " + filePath);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Calculates the effective resource values for SPark cluster managed by YARN.
	 * It considers the resource limits for scheduling containers by YARN
	 * and the need to fit an Application Master (AM) container in addition to the executor ones.
	 *
	 * @param memory total node memory inn bytes
	 * @param cores total node available virtual cores
	 * @param numExecutors number of available worker nodes
	 * @return arrays of length 5 -
	 * 		[executor mem. in MB, executor cores, num. executors, AM mem. in MB, AM cores]
	 */
	public static int[] getEffectiveExecutorResources(long memory, int cores, int numExecutors) {
		int effectiveExecutorMemoryMB, effectiveAmMemoryMB;
		int effectiveExecutorCores, effectiveAmCores;
		int effectiveNumExecutors;
		// YARN reserves 25% of the total memory for other resources (OS, node management, etc.)
		long yarnAllocationMemory = (long) (memory * 0.75);
		// plan for resource allocation for YARN Application Master (AM) container
		int totalExecutorCores = cores * numExecutors;
		// Scale with the cluster size growth to allow for allocating efficient AM resource
		int amMemoryMB = calculateAmMemoryMB(totalExecutorCores);
		int amMemoryOverheadMB = Math.max(384, (int) (amMemoryMB * 0.1)); // Spark default config
		long amTotalMemory = (long) (amMemoryMB + amMemoryOverheadMB) * 1024 * 1024;
		int amCores = calculateAmCores(totalExecutorCores);

		// decide if is more effective to launch AM alongside an executor or on a dedicated node
		// plan for executor memory overhead -> 10% of the executor memory (division by 1.1, always over 384MB)
		if (amTotalMemory * numExecutors >= yarnAllocationMemory) {
			// the case only for a large cluster of small instances
			// in this case dedicate a whole node for the AM
			effectiveExecutorMemoryMB = (int) Math.floor(yarnAllocationMemory / (1.1 * 1024 * 1024));
			effectiveExecutorCores = cores;
			// maximize the AM resource since no resource will be left for an executor
			effectiveAmMemoryMB = effectiveExecutorMemoryMB;
			effectiveAmCores = cores;
			effectiveNumExecutors = numExecutors - 1;
		} else {
			// in this case leave room in each worker node for executor + AM containers
			effectiveExecutorMemoryMB = (int) Math.floor((yarnAllocationMemory - amTotalMemory) / (1.1 * 1024 * 1024));
			effectiveExecutorCores = cores - amCores;
			effectiveAmMemoryMB = amMemoryMB;
			effectiveAmCores = amCores;
			effectiveNumExecutors = numExecutors;
		}

		// always 5 return values
		return new int[] {
				effectiveExecutorMemoryMB,
				effectiveExecutorCores,
				effectiveNumExecutors,
				effectiveAmMemoryMB,
				effectiveAmCores
		};
	}

	public static int calculateAmMemoryMB(int totalExecutorCores) {
		// 512MB base Application Master memory budget + 256MB for each 16 cores extra
		return 512 + (int) Math.floor((double) totalExecutorCores / 16) * 256;
	}

	public static int calculateAmCores(int totalExecutorCores) {
		// at least 1 core per 64 cores in cluster
		int scaledCores = (int) Math.ceil((totalExecutorCores) / 64.0);
		// cap to 8 cores for large clusters (cores > 512)
		return Math.min(8, scaledCores);
	}

	public static long calculateEffectiveDriverMemoryBudget(long driverMemory, int totalExecutorCores) {
		// 1GB Resource Manager memory budget + 256MB for each 16 cores extra
		int effectiveBudgetMB =  1024 + (int) Math.floor((double) totalExecutorCores / 16) * 256;
		long effectiveBudgetBytes = ((long) effectiveBudgetMB * 1024 * 1024);
		// validation if the memory is negative or insufficient is to be done separately
		// return value in bytes
		return Math.min((long) (driverMemory * JVM_MEMORY_FACTOR),
				driverMemory - effectiveBudgetBytes);
	}
}
