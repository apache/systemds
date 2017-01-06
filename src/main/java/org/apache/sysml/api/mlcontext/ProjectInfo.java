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

	public ProjectInfo() {
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
	 * Obtain all the properties from the manifest as a sorted map.
	 * 
	 * @return the manifest properties as a sorted map
	 */
	public SortedMap<String, String> properties() {
		return properties;
	}

}
