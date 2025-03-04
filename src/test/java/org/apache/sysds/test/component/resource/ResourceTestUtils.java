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

import org.apache.commons.configuration2.PropertiesConfiguration;
import org.apache.sysds.resource.CloudInstance;
import org.junit.Assert;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

import static org.apache.sysds.resource.CloudUtils.GBtoBytes;

public class ResourceTestUtils {
	public static final String DEFAULT_REGIONAL_PRICE_TABLE = "./scripts/resource/aws_regional_prices.csv";
	public static final String DEFAULT_INSTANCE_INFO_TABLE = "./scripts/resource/ec2_stats.csv";
	private static final String TEST_ARTIFACTS = "./src/test/scripts/component/resource/artifacts/";
	private static final String MINIAL_REGION_TABLE = "minimal_aws_regional_prices.csv";
	private static final String MINIAL_INFO_TABLE = "minimal_ec2_stats.csv";
	public static final String TEST_REGION;
	public static final double TEST_FEE_RATIO;
	public static final double TEST_STORAGE_PRICE;

	static {
		// ensure valid region table in artifacts and init test values
		try {
			List<String> lines = Files.readAllLines(getMinimalFeeTableFile().toPath());
			if (lines.size() > 1) {
				String valueLine = lines.get(1);
				String[] lineParts = valueLine.split(",");
				if (lineParts.length != 3) throw new IOException();
				TEST_REGION = lineParts[0];
				TEST_FEE_RATIO = Double.parseDouble(lineParts[1]);
				TEST_STORAGE_PRICE = Double.parseDouble(lineParts[2]);
			} else {
				throw new IOException();
			}
		} catch (IOException e) {
			throw new RuntimeException("Invalid testing region table file");
		}
	}
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

	public static File getMinimalFeeTableFile() {
		return new File(TEST_ARTIFACTS+MINIAL_REGION_TABLE);
	}

	public static File getMinimalInstanceInfoTableFile() {
		return new File(TEST_ARTIFACTS+MINIAL_INFO_TABLE);
	}

	public static File generateTmpDMLScript(String...scriptLines) throws IOException {
		File tmpFile = File.createTempFile("tmpScript", ".dml");
		List<String> lines = Arrays.stream(scriptLines).collect(Collectors.toList());
		Files.write(tmpFile.toPath(), lines);
		return tmpFile;
	}

	public static PropertiesConfiguration generateTestingOptionsRequired(String outputPath) {
		return generateOptionsRequired(
				TEST_REGION,
				TEST_ARTIFACTS+MINIAL_INFO_TABLE,
				TEST_ARTIFACTS+MINIAL_REGION_TABLE,
				outputPath);
	}

	public static PropertiesConfiguration generateOptionsRequired(
			String region,
			String infoTable,
			String regionTable,
			String outputFolder
	) {
		return generateOptions(region, infoTable, regionTable, outputFolder,
				null, null, null, null, null, null, null, null,
				null, null, null, null, null, null, null, null);
	}

	public static PropertiesConfiguration generateOptions(
			String region,
			String infoTable,
			String regionTable,
			String outputFolder,
			String localInputs,
			String enumeration,
			String optimizationFunction,
			String maxTime,
			String maxPrice,
			String cpuQuota,
			String minExecutors,
			String maxExecutors,
			String instanceFamilies,
			String instanceSizes,
			String stepSize,
			String exponentialBase,
			String useLargestEstimate,
			String useCpEstimates,
			String useBroadcasts,
			String useOutputs
	) {
		PropertiesConfiguration options = new PropertiesConfiguration();

		addToMapIfNotNull(options, "REGION", region);
		addToMapIfNotNull(options, "INFO_TABLE", infoTable);
		addToMapIfNotNull(options, "REGION_TABLE", regionTable);
		addToMapIfNotNull(options, "OUTPUT_FOLDER", outputFolder);
		addToMapIfNotNull(options, "LOCAL_INPUTS", localInputs);
		addToMapIfNotNull(options, "ENUMERATION", enumeration);
		addToMapIfNotNull(options, "OPTIMIZATION_FUNCTION", optimizationFunction);
		addToMapIfNotNull(options, "MAX_TIME", maxTime);
		addToMapIfNotNull(options, "MAX_PRICE", maxPrice);
		addToMapIfNotNull(options, "CPU_QUOTA", cpuQuota);
		addToMapIfNotNull(options, "MIN_EXECUTORS", minExecutors);
		addToMapIfNotNull(options, "MAX_EXECUTORS", maxExecutors);
		addToMapIfNotNull(options, "INSTANCE_FAMILIES", instanceFamilies);
		addToMapIfNotNull(options, "INSTANCE_SIZES", instanceSizes);
		addToMapIfNotNull(options, "STEP_SIZE", stepSize);
		addToMapIfNotNull(options, "EXPONENTIAL_BASE", exponentialBase);
		addToMapIfNotNull(options, "USE_LARGEST_ESTIMATE", useLargestEstimate);
		addToMapIfNotNull(options, "USE_CP_ESTIMATES", useCpEstimates);
		addToMapIfNotNull(options, "USE_BROADCASTS", useBroadcasts);
		addToMapIfNotNull(options, "USE_OUTPUTS", useOutputs);

		return options;
	}

	private static void addToMapIfNotNull(PropertiesConfiguration options, String key, String value) {
		if (value != null) {
			options.setProperty(key, value);
		}
	}

	public static void deleteDirectoryWithFiles(Path dir) throws IOException {
		// delete files in the directory and then the already empty directory itself
		Files.walkFileTree(dir, new SimpleFileVisitor<>() {
			@Override
			public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
				Files.delete(file);
				return FileVisitResult.CONTINUE;
			}

			@Override
			public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
				Files.delete(dir);
				return FileVisitResult.CONTINUE;
			}
		});
	}
}
