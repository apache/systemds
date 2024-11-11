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
import org.apache.sysds.resource.CloudUtils;
import org.apache.sysds.resource.CloudUtils.InstanceFamily;
import org.apache.sysds.resource.CloudUtils.InstanceSize;
import org.junit.Assert;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.HashMap;

import static org.apache.sysds.test.component.resource.ResourceTestUtils.*;
import static org.junit.Assert.*;

@net.jcip.annotations.NotThreadSafe
public class CloudUtilsTests {

	@Test
	public void getInstanceFamilyTest() {
		InstanceFamily expectedValue = InstanceFamily.M5;
		CloudUtils.InstanceFamily actualValue;

		actualValue = CloudUtils.getInstanceFamily("m5.xlarge");
		assertEquals(expectedValue, actualValue);

		actualValue = CloudUtils.getInstanceFamily("M5.XLARGE");
		assertEquals(expectedValue, actualValue);

		try {
			CloudUtils.getInstanceFamily("NON-M5.xlarge");
			fail("Throwing IllegalArgumentException was expected");
		} catch (IllegalArgumentException e) {
			// this block ensures correct execution of the test
		}
	}

	@Test
	public void getInstanceSizeTest() {
		InstanceSize expectedValue = InstanceSize._XLARGE;
		InstanceSize actualValue;

		actualValue = CloudUtils.getInstanceSize("m5.xlarge");
		assertEquals(expectedValue, actualValue);

		actualValue = CloudUtils.getInstanceSize("M5.XLARGE");
		assertEquals(expectedValue, actualValue);

		try {
			CloudUtils.getInstanceSize("m5.nonxlarge");
			fail("Throwing IllegalArgumentException was expected");
		} catch (IllegalArgumentException e) {
			// this block ensures correct execution of the test
		}
	}

	@Test
	public void validateInstanceNameTest() {
		// basic intel instance (old)
		assertTrue(CloudUtils.validateInstanceName("m5.2xlarge"));
		assertTrue(CloudUtils.validateInstanceName("M5.2XLARGE"));
		// basic intel instance (new)
		assertTrue(CloudUtils.validateInstanceName("m6i.xlarge"));
		// basic amd instance
		assertTrue(CloudUtils.validateInstanceName("m6a.xlarge"));
		// basic graviton instance
		assertTrue(CloudUtils.validateInstanceName("m6g.xlarge"));
		// invalid values
		assertFalse(CloudUtils.validateInstanceName("v5.xlarge"));
		assertFalse(CloudUtils.validateInstanceName("m5.notlarge"));
		assertFalse(CloudUtils.validateInstanceName("m5xlarge"));
		assertFalse(CloudUtils.validateInstanceName(".xlarge"));
		assertFalse(CloudUtils.validateInstanceName("m5."));
	}

	@Test
	public void loadDefaultFeeTableTest() {
		// test that the provided default file is accounted as valid by the function for loading
		String[] regions = {
				"us-east-1",
				"us-east-2",
				"us-west-1",
				"us-west-2",
				"ca-central-1",
				"ca-west-1",
				"af-south-1",
				"ap-east-1",
				"ap-south-2",
				"ap-southeast-3",
				"ap-southeast-5",
				"ap-southeast-4",
				"ap-south-1",
				"ap-northeast-3",
				"ap-northeast-2",
				"ap-southeast-1",
				"ap-southeast-2",
				"ap-northeast-1",
				"eu-central-1",
				"eu-west-1",
				"eu-west-2",
				"eu-south-1",
				"eu-west-3",
				"eu-south-2",
				"eu-north-1",
				"eu-central-2",
				"il-central-1",
				"me-south-1",
				"me-central-1",
				"sa-east-1"
		};

		for (String region : regions) {
			try {
				double[] prices = CloudUtils.loadRegionalPrices(DEFAULT_REGIONAL_PRICE_TABLE, region);
				double feeRatio = prices[0];
				double ebsPrice = prices[1];
				Assert.assertTrue(feeRatio >= 0.15 && feeRatio <= 0.25);
				Assert.assertTrue(ebsPrice >= 0.08);
			} catch (IOException e) {
				Assert.fail("Throwing IOException not expected: " + e);
			}
		}
	}

	@Test
	public void loadingInstanceInfoTest() throws IOException {
		// test the proper loading of the table
		File tmpFile = ResourceTestUtils.generateMinimalInstanceInfoTableFile();

		HashMap<String, CloudInstance> actual = CloudUtils.loadInstanceInfoTable(tmpFile.getPath(), TEST_FEE_RATIO, TEST_STORAGE_PRICE);
		HashMap<String, CloudInstance> expected = getSimpleCloudInstanceMap();

		for (String instanceName: expected.keySet()) {
			assertEqualsCloudInstances(expected.get(instanceName), actual.get(instanceName));
		}

		Files.deleteIfExists(tmpFile.toPath());
	}

	@Test
	public void loadDefaultInstanceInfoTableFile() throws IOException {
		// test that the provided default file is accounted as valid by the function for loading
		HashMap<String, CloudInstance> instanceMap = CloudUtils.loadInstanceInfoTable(DEFAULT_INSTANCE_INFO_TABLE, TEST_FEE_RATIO, TEST_STORAGE_PRICE);
		// test if all instances from 'M', 'C' or 'R' families
		// and if the minimum size is xlarge as required for EMR
		for (String instanceType : instanceMap.keySet()) {
			Assert.assertTrue(instanceType.startsWith("m") || instanceType.startsWith("c") || instanceType.startsWith("r"));
			Assert.assertTrue(instanceType.contains("xlarge"));
		}
	}
}
