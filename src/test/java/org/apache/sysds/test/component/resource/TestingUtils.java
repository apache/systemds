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

package org.apache.sysds.test.component.resource;

import org.apache.sysds.resource.CloudInstance;
import org.junit.Assert;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

import static org.apache.sysds.resource.CloudUtils.GBtoBytes;

public class TestingUtils {
	public static final String DEFAULT_REGIONAL_PRICE_TABLE = "./scripts/resource/aws_regional_prices.csv";
	public static final String DEFAULT_INSTANCE_INFO_TABLE = "./scripts/resource/ec2_stats.csv";
	public static final String TEST_REGION = "us-east-1";
	public static final double TEST_FEE_RATIO = 0.25;
	public static final double TEST_STORAGE_PRICE = 0.08;
	public static void assertEqualsCloudInstances(CloudInstance expected, CloudInstance actual) {
		Assert.assertEquals(expected.getInstanceName(), actual.getInstanceName());
		Assert.assertEquals(expected.getMemory(), actual.getMemory());
		Assert.assertEquals(expected.getVCPUs(), actual.getVCPUs());
		Assert.assertEquals(expected.getFLOPS(), actual.getFLOPS());
		Assert.assertEquals(expected.getMemoryBandwidth(), actual.getMemoryBandwidth(), 0.0);
		Assert.assertEquals(expected.getDiskReadBandwidth(), actual.getDiskReadBandwidth(), 0.0);
		Assert.assertEquals(expected.getDiskWriteBandwidth(), actual.getDiskWriteBandwidth(), 0.0);
		Assert.assertEquals(expected.getNetworkBandwidth(), actual.getNetworkBandwidth(), 0.0);
		Assert.assertEquals(expected.getPrice(), actual.getPrice(), 0.0);

	}

	public static HashMap<String, CloudInstance> getSimpleCloudInstanceMap() {
		HashMap<String, CloudInstance> instanceMap =  new HashMap<>();
		// fill the map wsearchStrategyh enough cloud instances to allow testing all search space dimension searchStrategyerations
		instanceMap.put("m5.xlarge", new CloudInstance("m5.xlarge", 0.192, TEST_FEE_RATIO*0.192, TEST_STORAGE_PRICE, GBtoBytes(16), 4, 160, 9934.166667, 143.72, 143.72, 156.25,false, 2, 32));
		instanceMap.put("m5.2xlarge", new CloudInstance("m5.2xlarge", 0.384, TEST_FEE_RATIO*0.384, TEST_STORAGE_PRICE, GBtoBytes(32), 8, 320, 19868.33333, 287.50, 287.50, 312.5, false, 4, 32));
		instanceMap.put("m5d.xlarge", new CloudInstance("m5d.xlarge", 0.226, TEST_FEE_RATIO * 0.226, TEST_STORAGE_PRICE, GBtoBytes(16), 4, 160, 9934.166667, 230.46875, 113.28125, 156.25, true, 1, 150));
		instanceMap.put("m5n.xlarge", new CloudInstance("m5n.xlarge", 0.238, TEST_FEE_RATIO * 0.238, TEST_STORAGE_PRICE, GBtoBytes(16), 4, 160, 9934.166667, 143.72, 143.72, 512.5, false, 2, 32));
		instanceMap.put("c5.xlarge", new CloudInstance("c5.xlarge", 0.17, TEST_FEE_RATIO*0.17, TEST_STORAGE_PRICE, GBtoBytes(8), 4, 192, 9934.166667, 143.72, 143.72, 156.25,false, 2, 32));
		instanceMap.put("c5d.xlarge", new CloudInstance("c5d.xlarge", 0.192, TEST_FEE_RATIO * 0.192, TEST_STORAGE_PRICE, GBtoBytes(8), 4, 192, 9934.166667, 163.84, 73.728, 156.25, true, 1, 100));
		instanceMap.put("c5n.xlarge", new CloudInstance("c5n.xlarge", 0.216, TEST_FEE_RATIO * 0.216, TEST_STORAGE_PRICE, GBtoBytes(10.5), 4, 192, 9934.166667, 143.72, 143.72, 625, false, 2, 32));
		instanceMap.put("c5.2xlarge", new CloudInstance("c5.2xlarge", 0.34, TEST_FEE_RATIO*0.34, TEST_STORAGE_PRICE, GBtoBytes(16), 8, 384, 19868.33333, 287.50, 287.50, 312.5, false, 4, 32));

		return instanceMap;
	}

	public static File generateMinimalFeeTableFile() throws IOException {
		File tmpFile = File.createTempFile("fee_tmp", ".csv");

		List<String> csvLines = Arrays.asList(
				"Region,Fee Ratio,EBS Price",
				String.format("%s,%.5f,%.5f", TEST_REGION, TEST_FEE_RATIO, TEST_STORAGE_PRICE)
		);
		Files.write(tmpFile.toPath(), csvLines);
		return tmpFile;
	}

	public static File generateMinimalInstanceInfoTableFile() throws IOException {
		File tmpFile = File.createTempFile("ec2_tmp", ".csv");
		// the minimal info table includes instances of different families, optimization purposes, sizes, storage classes and network classes
		List<String> csvLines = Arrays.asList(
				"API_Name,Price,Memory,vCPUs,Cores,GFLOPS,memBandwidth,NVMe,storageVolumes,sizePerVolume,readStorageBandwidth,writeStorageBandwidth,networkBandwidth",
				"m5.xlarge,0.1920000000,16.0,4,2,160,9934.166667,false,2,32,143.72,143.72,156.25",
				"m5d.xlarge,0.2260000000,16.0,4,2,160,9934.166667,true,1,150,230.46875,113.28125,156.25",
				"m5n.xlarge,0.2380000000,16,4,2,160,9934.166667,false,2,32,143.72,143.72,512.5",
				"m5.2xlarge,0.3840000000,32.0,8,4,320,19868.33333,false,4,32,287.5,287.5,312.5",
				"c5.xlarge,0.1700000000,8.0,4,2,192,9934.166667,false,2,32,143.72,143.72,156.25",
				"c5d.xlarge,0.1920000000,8.0,4,2,192,9934.166667,true,1,100,163.84,73.728,156.25",
				"c5n.xlarge,0.2160000000,10.5,4,2,192,9934.166667,false,2,32,143.72,143.72,625",
				"c5.2xlarge,0.3400000000,16.0,8,4,384,19868.33333,false,4,32,287.5,287.5,312.5"
		);
		Files.write(tmpFile.toPath(), csvLines);
		return tmpFile;
	}

	public static File generateTmlDMLScript(String...scriptLines) throws IOException {
		File tmpFile = File.createTempFile("tmpScript", ".dml");
		List<String> lines = Arrays.stream(scriptLines).collect(Collectors.toList());
		Files.write(tmpFile.toPath(), lines);
		return tmpFile;
	}
}
