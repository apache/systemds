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

package org.apache.sysml.api.mlcontext;

import java.io.IOException;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.jar.Attributes;
import java.util.jar.Attributes.Name;
import java.util.jar.JarFile;
import java.util.jar.Manifest;

/**
 * Obtains information that is stored in the manifest when the SystemML jar is
 * built.
 *
 */
public class ProjectInfo {

	SortedMap<String, String> properties = null;
	static ProjectInfo projectInfo = null;

	/**
	 * Return a ProjectInfo singleton instance.
	 *
	 * @return the ProjectInfo singleton instance
	 */
	public static ProjectInfo getProjectInfo() {
		if (projectInfo == null) {
			projectInfo = new ProjectInfo();
		}
		return projectInfo;
	}

	private ProjectInfo() {
		JarFile systemMlJar = null;
		try {
			String path = this.getClass().getProtectionDomain().getCodeSource().getLocation().getPath();
			systemMlJar = new JarFile(path);
			Manifest manifest = systemMlJar.getManifest();
			Attributes mainAttributes = manifest.getMainAttributes();
			properties = new TreeMap<String, String>();
			for (Object key : mainAttributes.keySet()) {
				String value = mainAttributes.getValue((Name) key);
				properties.put(key.toString(), value);
			}
		} catch (Exception e) {
			throw new MLContextException("Error trying to read from manifest in SystemML jar file", e);
		} finally {
			if (systemMlJar != null) {
				try {
					systemMlJar.close();
				} catch (IOException e) {
					throw new MLContextException("Error closing SystemML jar file", e);
				}
			}
		}
	}

	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		Set<String> keySet = properties.keySet();
		for (String key : keySet) {
			sb.append(key + ": " + properties.get(key) + "\n");
		}
		return sb.toString();
	}

	/**
	 * Obtain a manifest property value based on the key.
	 *
	 * @param key
	 *            the property key
	 * @return the property value
	 */
	public String property(String key) {
		return properties.get(key);
	}

	/**
	 * Obtain the project version from the manifest.
	 *
	 * @return the project version
	 */
	public String version() {
		return property("Version");
	}

	/**
	 * Object the artifact build time from the manifest.
	 *
	 * @return the artifact build time
	 */
	public String buildTime() {
		return property("Build-Time");
	}

	/**
	 * Obtain the minimum recommended Spark version from the manifest.
	 *
	 * @return the minimum recommended Spark version
	 */
	public String minimumRecommendedSparkVersion() {
		return property("Minimum-Recommended-Spark-Version");
	}

	/**
	 * Obtain all the properties from the manifest as a sorted map.
	 *
	 * @return the manifest properties as a sorted map
	 */
	public SortedMap<String, String> properties() {
		return properties;
	}

}
