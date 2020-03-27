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

package org.tugraz.sysds.api;

/**
 * This interface defines the programmatic access to dml configuration properties
 * (as defined in SystemDS-config.xml) to ensure API consistency across all APIs.
 */
public interface ConfigurableAPI 
{
	/**
	 * Reset configuration settings to default settings.
	 */
	public void resetConfig();
	
	/**
	 * Set configuration property, such as
	 * {@code setConfigProperty("sysds.localtmpdir", "/tmp/systemds")}.
	 *
	 * @param propertyName
	 *            property name
	 * @param propertyValue
	 *            property value
	 */
	public void setConfigProperty(String propertyName, String propertyValue);
}
