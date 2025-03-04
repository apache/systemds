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

import static org.apache.sysds.resource.CloudUtils.EBS_DEFAULT_ROOT_SIZE_EMR;

/**
 * This class describes the configurations of a single VM instance.
 * The idea is to use this class to represent instances of different
 * cloud hypervisors - currently supporting only EC2 instances by AWS.
 */
public class CloudInstance {
	private final String instanceName;
	// relevant for monetary costs
	private final double instancePrice; // per second
	private final double extraStoragePrice; // per GB per second
	private final double extraFee; // per second
	// relevant for compilation + time costs
	private final long memory; // in bytes
	private final int vCPUCores;
	// relevant for time costs
	private final long flops; // Float operations per second
	private final double memoryBandwidth; // in MB/s
	private final double diskReadBandwidth; // in MB/s
	private final double diskWriteBandwidth; // in MB/s
	private final double networkBandwidth; // in MB/s
	// relevant for final configurations only
	private final boolean NVMeStorage; // true: directly attached NVMe-based storage; false: network connected SSD storage
	private final int numberStorageVolumes;
	private final double sizeStorageVolumes; // in GB
	public CloudInstance(
			String instanceName,
			double pricePerHour,
			double extraFee,
			double storagePrice,
			long memory,
			int vCPUCores,
			double GFlops,
			double memoryBandwidth,
			double diskReadBandwidth,
			double diskWriteBandwidth,
			double networkBandwidth,
			boolean NVMeStorage,
			int numberStorageVolumes,
			double sizeStorageVolumes
	) {
		this.instanceName = instanceName;
		this.instancePrice = pricePerHour / 3600; // instance price per second
		this.extraFee = extraFee / 3600; // extra fee per second
		this.extraStoragePrice = NVMeStorage? EBS_DEFAULT_ROOT_SIZE_EMR : // no need of attaching extra
				storagePrice * (EBS_DEFAULT_ROOT_SIZE_EMR + numberStorageVolumes * sizeStorageVolumes) /
						(30 * 24 * 3600); // total storage price per second
		this.memory = memory;
		this.vCPUCores = vCPUCores;
		this.flops = (long) (GFlops*1024)*1024*1024;
		this.memoryBandwidth = memoryBandwidth;
		this.diskReadBandwidth = diskReadBandwidth;
		this.diskWriteBandwidth = diskWriteBandwidth;
		this.networkBandwidth = networkBandwidth;
		this.NVMeStorage = NVMeStorage;
		this.numberStorageVolumes = numberStorageVolumes;
		this.sizeStorageVolumes = sizeStorageVolumes;
	}

	public String getInstanceName() {
		return instanceName;
	}

	/**
	 * @return memory of the instance in B
	 */
	public long getMemory() {
		return memory;
	}

	/**
	 * @return number of virtual CPU cores of the instance
	 */
	public int getVCPUs() {
		return vCPUCores;
	}

	/**
	 * @return price per second of the instance
	 */
	public double getPrice() {
		return instancePrice;
	}

	/**
	 * @return price per second of whole extra storage (to be) attached
	 */
	public double getExtraStoragePrice() {
		return extraStoragePrice;
	}

	/**
	 * @return price per second for extra fee per instance (cluster management)
	 */
	public double getExtraFee() {
		return extraFee;
	}

	/**
	 * @return number of FLOPS of the instance
	 */
	public long getFLOPS() {
		return flops;
	}

	/**
	 * @return memory bandwidth of the instance in MB/s
	 */
	public double getMemoryBandwidth() {
		return memoryBandwidth;
	}

	/**
	 * @return disk read bandwidth of the instance in MB/s
	 */
	public double getDiskReadBandwidth() {
		return diskReadBandwidth;
	}

	/**
	 * @return disk write bandwidth of the instance in MB/s
	 */
	public double getDiskWriteBandwidth() {
		return diskWriteBandwidth;
	}

	/**
	 * @return network bandwidth of the instance in MB/s
	 */
	public double getNetworkBandwidth() {
		return networkBandwidth;
	}

	/**
	 * @return flag if the instance has attached NVMe-based storage volume(s)
	 */
	public boolean isNVMeStorage() { return NVMeStorage; }

	/**
	 * @return number of storage volumes (attached or to be configured)
	 */
	public int getNumStorageVolumes() { return numberStorageVolumes; }

	/**
	 * @return size of each storage volume (all equally sized) in GB
	 */
	public double getSizeStoragePerVolume() { return sizeStorageVolumes; }
}
