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

public class AWSUtils extends CloudUtils {
	public static final String EC2_REGEX = "^([a-z]+)([0-9])(a|g|i?)([bdnez]*)\\.([a-z0-9]+)$";
	@Override
	public boolean validateInstanceName(String input) {
		String instanceName = input.toLowerCase();
		if (!instanceName.toLowerCase().matches(EC2_REGEX)) return false;
		try {
			getInstanceType(instanceName);
			getInstanceSize(instanceName);
		} catch (IllegalArgumentException e) {
			return false;
		}
		return true;
	}

	@Override
	public InstanceType getInstanceType(String instanceName) {
		String typeAsString = instanceName.split("\\.")[0];
		// throws exception if string value is not valid
		return InstanceType.customValueOf(typeAsString);
	}

	@Override
	public InstanceSize getInstanceSize(String instanceName) {
		String sizeAsString = instanceName.split("\\.")[1];
		// throws exception if string value is not valid
		return InstanceSize.customValueOf(sizeAsString);
	}

	@Override
	public double calculateClusterPrice(EnumerationUtils.ConfigurationPoint config, double time) {
		double pricePerSeconds = getClusterCostPerHour(config) / 3600;
		return time * pricePerSeconds;
	}

	private double getClusterCostPerHour(EnumerationUtils.ConfigurationPoint config) {
		if (config.numberExecutors == 0) {
			return config.driverInstance.getPrice();
		}
		return config.driverInstance.getPrice() +
				config.executorInstance.getPrice()*config.numberExecutors;
	}
}
