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

package org.apache.sysds.api.mlcontext;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysds.runtime.io.IOUtilFunctions;

/**
 * Factory for creating DML and PYDML Script objects from strings, files, URLs,
 * and input streams.
 *
 */
public class ScriptFactory {

	/**
	 * Create a DML Script object based on a string path to a file.
	 *
	 * @param scriptFilePath
	 *            path to DML script file (local or HDFS)
	 * @return DML Script object
	 */
	public static Script dmlFromFile(String scriptFilePath) {
		return scriptFromFile(scriptFilePath);
	}

	/**
	 * Create a DML Script object based on an input stream.
	 *
	 * @param inputStream
	 *            input stream to DML
	 * @return DML Script object
	 */
	public static Script dmlFromInputStream(InputStream inputStream) {
		return scriptFromInputStream(inputStream);
	}

	/**
	 * Creates a DML Script object based on a file in the local file system. To
	 * create a DML Script object from a local file or HDFS, please use
	 * {@link #dmlFromFile(String)}.
	 *
	 * @param localScriptFile
	 *            the local DML file
	 * @return DML Script object
	 */
	public static Script dmlFromLocalFile(File localScriptFile) {
		return scriptFromLocalFile(localScriptFile);
	}

	/**
	 * Create a DML Script object based on a string.
	 *
	 * @param scriptString
	 *            string of DML
	 * @return DML Script object
	 */
	public static Script dmlFromString(String scriptString) {
		return scriptFromString(scriptString);
	}

	/**
	 * Create a DML Script object based on a URL path.
	 *
	 * @param scriptUrlPath
	 *            URL path to DML script
	 * @return DML Script object
	 */
	public static Script dmlFromUrl(String scriptUrlPath) {
		return scriptFromUrl(scriptUrlPath);
	}

	/**
	 * Create a DML Script object based on a URL.
	 *
	 * @param scriptUrl
	 *            URL to DML script
	 * @return DML Script object
	 */
	public static Script dmlFromUrl(URL scriptUrl) {
		return scriptFromUrl(scriptUrl);
	}

	/**
	 * Create a DML Script object based on a resource path.
	 *
	 * @param resourcePath
	 *            path to a resource on the classpath
	 * @return DML Script object
	 */
	public static Script dmlFromResource(String resourcePath) {
		return scriptFromResource(resourcePath);
	}

	/**
	 * Create a DML or PYDML Script object based on a string path to a file.
	 *
	 * @param scriptFilePath
	 *            path to DML or PYDML script file (local or HDFS)
	 * @param scriptType
	 *            {@code ScriptType.DML} or {@code ScriptType.PYDML}
	 * @return DML or PYDML Script object
	 */
	private static Script scriptFromFile(String scriptFilePath) {
		String scriptString = getScriptStringFromFile(scriptFilePath);
		return scriptFromString(scriptString).setName(scriptFilePath);
	}

	/**
	 * Create a DML or PYDML Script object based on an input stream.
	 *
	 * @param inputStream
	 *            input stream to DML or PYDML
	 * @param scriptType
	 *            {@code ScriptType.DML} or {@code ScriptType.PYDML}
	 * @return DML or PYDML Script object
	 */
	private static Script scriptFromInputStream(InputStream inputStream) {
		String scriptString = getScriptStringFromInputStream(inputStream);
		return scriptFromString(scriptString);
	}

	/**
	 * Creates a DML or PYDML Script object based on a file in the local file
	 * system. To create a Script object from a local file or HDFS, please use
	 * {@link scriptFromFile(String, ScriptType)}.
	 *
	 * @param localScriptFile
	 *            The local DML or PYDML file
	 * @param scriptType
	 *            {@code ScriptType.DML} or {@code ScriptType.PYDML}
	 * @return DML or PYDML Script object
	 */
	private static Script scriptFromLocalFile(File localScriptFile) {
		String scriptString = getScriptStringFromFile(localScriptFile);
		return scriptFromString(scriptString).setName(localScriptFile.getName());
	}

	/**
	 * Create a DML or PYDML Script object based on a string.
	 *
	 * @param scriptString
	 *            string of DML or PYDML
	 * @return DML or PYDML Script object
	 */
	private static Script scriptFromString(String scriptString) {
		return new Script(scriptString);
	}

	/**
	 * Creat a DML or PYDML Script object based on a URL path.
	 *
	 * @param scriptUrlPath
	 *            URL path to DML or PYDML script
	 * @return DML or PYDML Script object
	 */
	private static Script scriptFromUrl(String scriptUrlPath) {
		String scriptString = getScriptStringFromUrl(scriptUrlPath);
		return scriptFromString(scriptString).setName(scriptUrlPath);
	}

	/**
	 * Create a DML or PYDML Script object based on a URL.
	 *
	 * @param scriptUrl
	 *            URL to DML or PYDML script
	 * @return DML or PYDML Script object
	 */
	private static Script scriptFromUrl(URL scriptUrl) {
		String scriptString = getScriptStringFromUrl(scriptUrl);
		return scriptFromString(scriptString).setName(scriptUrl.toString());
	}

	/**
	 * Create a DML or PYDML Script object based on a resource path.
	 *
	 * @param resourcePath
	 *            path to a resource on the classpath
	 * @return DML or PYDML Script object
	 */
	private static Script scriptFromResource(String resourcePath) {
		if (resourcePath == null) {
			return null;
		}
		if (!resourcePath.startsWith("/")) {
			resourcePath = "/" + resourcePath;
		}
		try( InputStream inputStream = ScriptFactory.class.getResourceAsStream(resourcePath) ) {
			return scriptFromInputStream(inputStream).setName(resourcePath);
		} catch (Exception e){
			throw new MLContextException("Error trying to read script from resource: "+ resourcePath, e);
		}
	}

	/**
	 * Create a DML Script object based on a string.
	 *
	 * @param scriptString
	 *            string of DML
	 * @return DML Script object
	 */
	public static Script dml(String scriptString) {
		return dmlFromString(scriptString);
	}

	/**
	 * Obtain a script string from a file in the local file system. To obtain a
	 * script string from a file in HDFS, please use
	 * getScriptStringFromFile(String scriptFilePath).
	 *
	 * @param file
	 *            The script file.
	 * @return The script string.
	 * @throws MLContextException
	 *             If a problem occurs reading the script string from the file.
	 */
	private static String getScriptStringFromFile(File file) {
		if (file == null) {
			throw new MLContextException("Script file is null");
		}
		String filePath = file.getPath();
		try {
			return FileUtils.readFileToString(file);
		} catch (IOException e) {
			throw new MLContextException("Error trying to read script string from file: " + filePath, e);
		}
	}

	/**
	 * Obtain a script string from a file.
	 *
	 * @param scriptFilePath
	 *            The file path to the script file (either local file system or
	 *            HDFS)
	 * @return The script string
	 * @throws MLContextException
	 *             If a problem occurs reading the script string from the file
	 */
	private static String getScriptStringFromFile(String scriptFilePath) {
		if (scriptFilePath == null) {
			throw new MLContextException("Script file path is null");
		}
		try {
			if (   scriptFilePath.startsWith("hdfs:") || scriptFilePath.startsWith("gpfs:")
				|| IOUtilFunctions.isObjectStoreFileScheme(new Path(scriptFilePath))) {
				Path path = new Path(scriptFilePath);
				FileSystem fs = IOUtilFunctions.getFileSystem(path);
				try( FSDataInputStream fsdis = fs.open(path) ) {
					return IOUtils.toString(fsdis);
				}
			}
			// from local file system
			File scriptFile = new File(scriptFilePath);
			return FileUtils.readFileToString(scriptFile);
		}
		catch (IllegalArgumentException | IOException e) {
			throw new MLContextException("Error trying to read script string from file: " + scriptFilePath, e);
		}
	}

	/**
	 * Obtain a script string from an InputStream.
	 *
	 * @param inputStream
	 *            The InputStream from which to read the script string
	 * @return The script string
	 * @throws MLContextException
	 *             If a problem occurs reading the script string from the URL
	 */
	private static String getScriptStringFromInputStream(InputStream inputStream) {
		if (inputStream == null) {
			throw new MLContextException("InputStream is null");
		}
		try {
			return IOUtils.toString(inputStream);
		} catch (IOException e) {
			throw new MLContextException("Error trying to read script string from InputStream", e);
		}
	}

	/**
	 * Obtain a script string from a URL.
	 *
	 * @param scriptUrlPath
	 *            The URL path to the script file
	 * @return The script string
	 * @throws MLContextException
	 *             If a problem occurs reading the script string from the URL
	 */
	private static String getScriptStringFromUrl(String scriptUrlPath) {
		if (scriptUrlPath == null) {
			throw new MLContextException("Script URL path is null");
		}
		try {
			URL url = new URL(scriptUrlPath);
			return getScriptStringFromUrl(url);
		} catch (MalformedURLException e) {
			throw new MLContextException("Error trying to read script string from URL path: " + scriptUrlPath, e);
		}
	}

	/**
	 * Obtain a script string from a URL.
	 *
	 * @param url
	 *            The script URL
	 * @return The script string
	 * @throws MLContextException
	 *             If a problem occurs reading the script string from the URL
	 */
	private static String getScriptStringFromUrl(URL url) {
		if (url == null) {
			throw new MLContextException("URL is null");
		}
		String urlString = url.toString();
		if ((!urlString.toLowerCase().startsWith("http:")) && (!urlString.toLowerCase().startsWith("https:"))) {
			throw new MLContextException("Currently only reading from http and https URLs is supported");
		}
		try( InputStream is = url.openStream() ) {
			return IOUtils.toString(is);
		} catch (IOException e) {
			throw new MLContextException("Error trying to read script string from URL: " + url, e);
		}
	}
}
