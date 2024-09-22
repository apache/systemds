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

import static org.apache.sysds.resource.CloudUtils.GBtoBytes;

public class TestingUtils {
	public static void assertEqualsCloudInstances(CloudInstance expected, CloudInstance actual) {
		Assert.assertEquals(expected.getInstanceName(), actual.getInstanceName());
		Assert.assertEquals(expected.getMemory(), actual.getMemory());
		Assert.assertEquals(expected.getVCPUs(), actual.getVCPUs());
		Assert.assertEquals(expected.getFLOPS(), actual.getFLOPS());
		Assert.assertEquals(expected.getMemorySpeed(), actual.getMemorySpeed(), 0.0);
		Assert.assertEquals(expected.getDiskSpeed(), actual.getDiskSpeed(), 0.0);
		Assert.assertEquals(expected.getNetworkSpeed(), actual.getNetworkSpeed(), 0.0);
		Assert.assertEquals(expected.getPrice(), actual.getPrice(), 0.0);

	}

	public static HashMap<String, CloudInstance> getSimpleCloudInstanceMap() {
		HashMap<String, CloudInstance> instanceMap =  new HashMap<>();
		// fill the map wsearchStrategyh enough cloud instances to allow testing all search space dimension searchStrategyerations
		instanceMap.put("m5.xlarge", new CloudInstance("m5.xlarge", GBtoBytes(16), 4, 0.34375, 21328.0, 143.75, 160.0, 0.23));
		instanceMap.put("m5.2xlarge", new CloudInstance("m5.2xlarge", GBtoBytes(32), 8, 0.6875, 21328.0, 287.50, 320.0, 0.46));
		instanceMap.put("c5.xlarge", new CloudInstance("c5.xlarge", GBtoBytes(8), 4, 0.46875, 21328.0, 143.75, 160.0, 0.194));
		instanceMap.put("c5.2xlarge", new CloudInstance("c5.2xlarge", GBtoBytes(16), 8, 0.9375, 21328.0, 287.50, 320.0, 0.388));

		return instanceMap;
	}

	public static File generateTmpInstanceInfoTableFile() throws IOException {
		File tmpFile = File.createTempFile("systemds_tmp", ".csv");

		List<String> csvLines = Arrays.asList(
				"API_Name,Memory,vCPUs,gFlops,ramSpeed,diskSpeed,networkSpeed,Price",
				"m5.xlarge,16.0,4,0.34375,21328.0,143.75,160.0,0.23",
				"m5.2xlarge,32.0,8,0.6875,21328.0,287.50,320.0,0.46",
				"c5.xlarge,8.0,4,0.46875,21328.0,143.75,160.0,0.194",
				"c5.2xlarge,16.0,8,0.9375,21328.0,287.50,320.0,0.388"
		);
		Files.write(tmpFile.toPath(), csvLines);
		return tmpFile;
	}
}
