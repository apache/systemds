/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.validation;

import java.io.BufferedReader;
import java.io.BufferedOutputStream;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.FileUtils;


/**
 * Checks that all jar files included in the distribution are mentioned in the LICENSE file
 * and that all jars mentioned in the LICENSE are in the distribution.
 */
public class ValidateLicAndNotice 
{
	public static String[] fileTypes = {Constants.JAR, Constants.DLL, Constants.EXP, Constants.LIB, Constants.PDB, Constants.EXE};

	// Zip Distribution directory.
	private String strDistroDir =  "../../../target/release/systemds/target/";

	static final String[][] packageLicenses =
			{		{"org/antlr", "ANTLR 4 Runtime (http://www.antlr.org/antlr4-runtime) org.antlr:antlr4-runtime:4.5.3"},
					{"org/apache/wink/json4j","Apache Wink :: JSON4J (http://www.apache.org/wink/wink-json4j/) org.apache.wink:wink-json4j:1.4"},
					{"caffe","The proto file (src/main/proto/caffe/caffe.proto) is part of Caffe project,"},
					{"org/tensorflow","The proto files (src/main/proto/tensorflow/event.proto and src/main/proto/tensorflow/summary.proto) is part of TensorFlow project,"},
					{"jcuda","JCuda (jcuda.org)"},
			};

	public static HashMap<String, String[][]> hmJSLicenses = new HashMap<String, String[][]>();
	static {

		String [][] strTemp1 = {{"Bootstrap v3.3.6", "Copyright (c) 2011-2015 Twitter, Inc.", "false"}};
		hmJSLicenses.put("bootstrap.min.js", strTemp1);
		String [][] strTemp2 = {{"Normalize v3.0.3", "Copyright (c) Nicolas Gallagher and Jonathan Neal", "false"}};
		hmJSLicenses.put("bootstrap.min.css", strTemp2);
		String [][] strTemp3 = {{"AnchorJS v1.1.1", "Copyright (c) 2015 Bryan Braun", "false"}};
		hmJSLicenses.put("anchor.min.js", strTemp3);
		String [][] strTemp4 = {{"jQuery v1.12.0", "(c) jQuery Foundation", "false"},
								{"jQuery v1.12.0", "Copyright jQuery Foundation and other contributors, https://jquery.org/", "false"}};
		hmJSLicenses.put("jquery-1.12.0.min.js", strTemp4);
		String [][] strTemp5 = {{"Pygments", "Copyright (c) 2006-2017 by the respective authors (see AUTHORS file).", "false"}};
		hmJSLicenses.put("pygments-default.css", strTemp5);
	}

	public ValidateLicAndNotice() {
	}

	public ValidateLicAndNotice(String strDistroDir) {
		setDistroDir(strDistroDir);
	}

	public String getDistroDir() {
       		return strDistroDir;
	}

	public void setDistroDir(String strDistroDir) {
		this.strDistroDir = strDistroDir;
	}

	/**
	 * This will validate all archives from distribution location.
	 *
	 * @return Returns the output code
	 */
	public int validate() throws Exception {

		int retCode = Constants.SUCCESS, retCodeForAllFileTypes = Constants.SUCCESS, retCodeAll = Constants.SUCCESS;

		File distroRoot = new File( getDistroDir());
		File libDirectory = distroRoot;
		if (!libDirectory.exists()) {
			Utility.debugPrint(Constants.DEBUG_ERROR, "Distribution folder '" + libDirectory.getAbsoluteFile().toString() + "' does not exist.");
			return Constants.NO_ZIP_TGZ;
		}

		File outTempDir = File.createTempFile("outTemp", "");
		outTempDir.delete();
		outTempDir.mkdir();

		List<String> zips = getZipsInDistro(libDirectory);
		if(zips.size() == 0) {
			Utility.debugPrint(Constants.DEBUG_ERROR, "Can't find archives in folder: " + libDirectory.getAbsoluteFile().toString());
			return Constants.NO_ZIP_TGZ;
		}

		for (String zipFile: zips)
		{
			retCodeForAllFileTypes = Constants.SUCCESS;
			Utility.debugPrint(Constants.DEBUG_INFO, "======================================================================================");
			Utility.debugPrint(Constants.DEBUG_INFO, "Validating archive: " + zipFile + " ...");

			for (String fileType: fileTypes) {
				retCode = Constants.SUCCESS;

				List<String> filesAll = null;
				// Extract license/notice only at first time in all filetypes validation for a given zip.
				if(fileType == Constants.JAR) {
					if (!ValidateLicAndNotice.extractFile(libDirectory + "/" + zipFile, Constants.LICENSE, outTempDir.getAbsolutePath(), true))
						return Constants.FAILED_TO_EXTRACT;
					if (!ValidateLicAndNotice.extractFile(libDirectory + "/" + zipFile, Constants.NOTICE, outTempDir.getAbsolutePath(), true))
						return Constants.FAILED_TO_EXTRACT;
				}

				filesAll = getFiles(libDirectory + "/" + zipFile, fileType);

				File licenseFile = new File(outTempDir, Constants.LICENSE);
				List<String> files = new ArrayList<String>();
				List<String> fileSysds = new ArrayList<String>();
				for (String file : filesAll) {
					int sysdsLen = Constants.SYSTEMDS_NAME.length();
					String strBegPart = file.substring(0, sysdsLen);
					if (strBegPart.compareToIgnoreCase(Constants.SYSTEMDS_NAME) != 0)
						files.add(file);
					else
						fileSysds.add(file);
				}


				List<String> bad2 = getLICENSEFilesNotInList(licenseFile, files, fileType);
				if (bad2.size() > 0) {
					Utility.debugPrint(Constants.DEBUG_WARNING,"Files in LICENSE but not in Distribution: " + bad2);
					retCode += Constants.FILE_NOT_IN_ZIP;
				}

				List<String> bad1 = getFilesNotInLICENSE(licenseFile, files, fileType);
				if (bad1.size() > 0) {
					Utility.debugPrint(Constants.DEBUG_ERROR,"Files in distribution but not in LICENSE: " + bad1);
					retCode += Constants.FILE_NOT_IN_LIC;
				}

				// Validate shaded jar and notice only one time for each archive.
				if(fileType == Constants.JAR) {
					for (String file : fileSysds)
						retCode += ValidateLicAndNotice.validateShadedLic(libDirectory + "/" + zipFile, file, outTempDir.getAbsolutePath());
					if (!validateNotice(outTempDir.getAbsolutePath()+"/"+Constants.NOTICE)) {
						Utility.debugPrint(Constants.DEBUG_ERROR, "Notice validation failed, please check notice file manually in this archive.");
						retCode += Constants.INVALID_NOTICE;
					}
					if (!validateJSCssLicense(licenseFile, libDirectory + "/" + zipFile)) {
						Utility.debugPrint(Constants.DEBUG_ERROR, "JS/CSS license validation failed, please check license file manually in this archive.");
						retCode += Constants.JS_CSS_LIC_NOT_EXIST;
					}
				}

				if (retCode  == Constants.SUCCESS)
					Utility.debugPrint(Constants.DEBUG_INFO3, "Validation of file type '." + fileType + "' in archive " + zipFile + " completed successfully.");
				else {
					Utility.debugPrint(Constants.DEBUG_ERROR, "License/Notice validation failed for archive " + zipFile + " with error code " + retCode + ", please validate file manually.");
					retCodeForAllFileTypes = Constants.FAILURE;
				}
			}
			if(retCodeForAllFileTypes == Constants.SUCCESS)
				Utility.debugPrint(Constants.DEBUG_INFO, "Validation of archive " + zipFile + " completed successfully.");

			retCodeAll = retCodeForAllFileTypes != Constants.SUCCESS?Constants.FAILURE:retCodeAll;
		}
		Utility.debugPrint(Constants.DEBUG_INFO, "======================================================================================");

		FileUtils.deleteDirectory(outTempDir);
		return retCodeAll;
	}

	/**
	 * This will validate objects (class files) from jar file within a zip file.
	 *
	 * @param	zipFileName is the name of zip file from which set of class packages will be returned.
	 * @param 	fileName is the name of the file within zip (jar) file to validate list of packages within.
	 * @param 	outTempDir is the temporary directory name.
	 * @return  Success or Failure code
	 */
	public static int validateShadedLic(String zipFileName, String file, String outTempDir) throws Exception
	{

		File outTempDir2 = new File (outTempDir + "/" + "2");
		outTempDir2.mkdir();
		if(!ValidateLicAndNotice.extractFile(zipFileName, file, outTempDir2.getAbsolutePath(), false))
			return Constants.FAILED_TO_EXTRACT;
		if(!ValidateLicAndNotice.extractFile(outTempDir2.getAbsolutePath()+"/"+file, Constants.LICENSE, outTempDir2.getAbsolutePath(), true))
			return Constants.FAILED_TO_EXTRACT;

		HashMap<String, Boolean> hashMapPackages = getPackagesFromZip(outTempDir2.getAbsolutePath() + "/" + file);
		for (String packageName: hashMapPackages.keySet())
			Utility.debugPrint(Constants.DEBUG_CODE, "Package: " + packageName + " Licensed: " + hashMapPackages.get(packageName));

		int iRetCode = ValidateLicAndNotice.validatePackages(outTempDir2.getAbsolutePath()+"/"+Constants.LICENSE, hashMapPackages);

		FileUtils.deleteDirectory(outTempDir2);
		return iRetCode;
	}

	/**
	 * This will validate objects (class files) against license file.
	 *
	 * @param	licenseFile is the name of the license file.
	 * @param 	hashMapPackages is the list of package names to be validated for license.
	 * @return  Success or Failure code
	 */
	public static int validatePackages(String licenseFile, HashMap<String, Boolean> hashMapPackages) throws Exception
	{
		int iRetCode = Constants.SUCCESS;
		BufferedReader reader = new BufferedReader(new FileReader(licenseFile));
		String line = null;
		HashSet <String> packageValidLic = new HashSet<String>();
		while ((line = reader.readLine()) != null) {
			line = line.trim();
			for(int i=0; i <packageLicenses.length; ++i) {
				if (line.contains(packageLicenses[i][1])) {
					packageValidLic.add(packageLicenses[i][0]);
					Utility.debugPrint(Constants.DEBUG_INFO3, "License for package " + packageLicenses[i][0] + " exists.");
				}
			}
		}

		Iterator<Map.Entry<String, Boolean>> itPackages = hashMapPackages.entrySet().iterator();
		while (itPackages.hasNext()) {
			Map.Entry<String, Boolean> pairPackage = (Map.Entry<String, Boolean>) itPackages.next();
			Iterator<String> itLicPackages = packageValidLic.iterator();
			while (itLicPackages.hasNext()) {
				if(((String)pairPackage.getKey()).startsWith((String)itLicPackages.next()))
					pairPackage.setValue(Boolean.TRUE);
			}
		}

		itPackages = hashMapPackages.entrySet().iterator();
		while (itPackages.hasNext()) {
			Map.Entry pairPackage = (Map.Entry) itPackages.next();
			if(!(Boolean)pairPackage.getValue()) {
				Utility.debugPrint(Constants.DEBUG_WARNING, "Could not validate license for package " + pairPackage.getKey() + ", please validate manually.");
				iRetCode = Constants.LIC_NOT_EXIST;
			}
		}

		return iRetCode;
	}

	/**
	 * This will return the set of packages from zip file.
	 *
	 * @param	zipFileName is the name of zip file from which set of class packages will be returned.
	 * @return	Returns set of packages for classes included in the zip file .
	 */
	public static HashMap<String, Boolean> getPackagesFromZip (String zipFileName) throws Exception{
		HashMap<String, Boolean> packages = new HashMap<String, Boolean>();
		try {
			ZipEntry entry;
			ZipFile zipfile = new ZipFile(zipFileName);
			Enumeration e = zipfile.entries();
			while(e.hasMoreElements()) {
				entry = (ZipEntry) e.nextElement();
				if(! entry.getName().startsWith(Constants.SYSTEMDS_PACKAGE) &&
				     entry.getName().endsWith("." + Constants.CLASS)) {
					int iPos = entry.getName().lastIndexOf("/");
					if (iPos > 0) {
						String strPackageName = entry.getName().substring(0, iPos);
						packages.put(strPackageName, Boolean.FALSE);
						Utility.debugPrint(Constants.DEBUG_CODE, "Package found : " + strPackageName);
					}
				}
			}
		} catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
		return packages;
	}

	/**
	 * This will return the list of files in license files but not in list of files coming from archive.
	 *
	 * @param	licenseFile is the file against which contents of archive gets compared.
	 * @param 	files	are the list of files coming from archive.
	 * @param 	fileExt	is the extension of file to validate (e.g. "jar")
	 * @return 	Returns the list of files in License file but not in archive.
	 */
	private List<String> getLICENSEFilesNotInList(File licenseFile, List<String> files, String fileExt) throws IOException {

		List<String> badFiles = new ArrayList<String>();
		BufferedReader reader = new BufferedReader(new FileReader(licenseFile));
		String line = null;
		while ((line = reader.readLine()) != null) {
			line = line.trim();
			if (line.contains("." + fileExt)) {
				StringTokenizer st = new StringTokenizer(line);
				while (st.hasMoreTokens()) {
					String s = st.nextToken();
					if (s.contains("." + fileExt)) {
						if (s.startsWith("(")) {
							s = s.substring(1);
						}
						if (s.endsWith(",") || s.endsWith(":")) {
							s = s.substring(0, s.length()-1);
						}
						if (s.endsWith(")")) {
							s = s.substring(0, s.length()-1);
						}
						if (!files.contains(s)) {
							badFiles.add(s);
						}
					}
				}
			}
		}
		return badFiles;
	}

	/**
	 * This will return the list of files in license files with specified file extension.
	 *
	 * @param	licenseFile is the file against which contents of archive gets compared.
	 * @param 	fileExt	is the extension of file to validate (e.g. "jar")
	 * @return 	Returns the list of files in License file.
	 */
	private List<String> getFilesFromLicenseFile(File licenseFile, String fileExt) throws IOException {

		List<String> files = new ArrayList<String>();
		BufferedReader reader = new BufferedReader(new FileReader(licenseFile));
		String line = null;
		while ((line = reader.readLine()) != null) {
			line = line.trim();
			if (line.contains("." + fileExt)) {
				StringTokenizer st = new StringTokenizer(line);
				while (st.hasMoreTokens()) {
					String s = st.nextToken();
					if (s.contains("." + fileExt)) {
						if (s.startsWith("(")) {
							s = s.substring(1);
						}
						if (s.endsWith(",") || s.endsWith(":")) {
							s = s.substring(0, s.length()-1);
						}
						if (s.endsWith(")")) {
							s = s.substring(0, s.length()-1);
						}
						if (!files.contains(s)) {
							files.add(s);
						}
					}
				}
			}
		}
		return files;
	}

	/**
	* This will return the list of files coming from archive but not in the license file.
	 *
	 * @param	licenseFile is the file against which contents of archive gets compared.
	 * @param 	files	are the list of files coming from archive.
	 * @param 	fileExt	is the extension of file to validate (e.g. "jar")
	 * @return 	Returns the list of files in archive but not in License file.
	 */
	private List<String> getFilesNotInLICENSE(File licenseFile, List<String> files, String fileExt) throws IOException {
		List<String> badFiles = new ArrayList<String>();
		List<String> licFiles = getFilesFromLicenseFile(licenseFile, fileExt);
		for (String file : files) {
			if (!licFiles.contains(file)) {
				badFiles.add(file);
			}
		}
		return badFiles;
	}

	/**
	 * This will return the list of archives from a directory.
	 *
	 * @param	directory is the location from where list of archives will be returned.
	 * @return 	Returns the list of archives (e.g., .zip/tgz/tar.gz files) from a directory.
	 */
	private List<String> getZipsInDistro(File directory) {
		List<String> zips = new ArrayList<String>();
		for (String fileName : directory.list())
			if ((fileName.endsWith("." + Constants.ZIP)) || (fileName.endsWith("." + Constants.TGZ)) ||
				(fileName.endsWith("." + Constants.TAR_GZ))) {
				zips.add(fileName);
			}
		return zips;
	}

	/**
	 * This will return the content of file.
	 *
	 * @param	file is the parameter of type File from which contents will be read and returned.
	 * @return 	Returns the contents from file in String format.
	 */
	private static String readFile(File file) throws java.io.IOException {
		StringBuffer fileData = new StringBuffer();
	        BufferedReader reader = new BufferedReader(new FileReader(file));
		char[] buf = new char[1024];
		int numRead = 0;
		while ((numRead = reader.read(buf)) != -1) {
			String readData = String.valueOf(buf, 0, numRead);
			fileData.append(readData);
			buf = new char[1024];
		}
		reader.close();
		return fileData.toString();
	}

	/**
	 * This will return the file from archive and store it in specified location.
	 *
	 * @param	zipFileName is the name of archive from which file to be extracted.
	 * @param	fileName is the name of the file to be extracted.
	 * @param	strDestLoc is the location where file will be extracted.
	 * @param 	bFirstDirLevel to indicate to get file from first directory level.
	 * @return  Success or Failure
	 */
	public static boolean extractFile(String zipFileName, String fileName, String strDestLoc, boolean bFirstDirLevel) {
		Utility.debugPrint(Constants.DEBUG_CODE, "Extracting " + fileName + " from archive " + zipFileName);
		if (zipFileName.endsWith("." + Constants.ZIP) || zipFileName.endsWith("." + Constants.JAR))
			return extractFileFromZip(zipFileName, fileName, strDestLoc, bFirstDirLevel);
		else if (zipFileName.endsWith("." + Constants.TGZ) || zipFileName.endsWith("." + Constants.TAR_GZ))
			return extractFileFromTGZ(zipFileName, fileName, strDestLoc, bFirstDirLevel);
		return Constants.bFAILURE;
	}


	/**
	 * This will return the file from zip file and store it in specified location.
	 *
	 * @param	zipFileName is the name of zip file from which file to be extracted.
	 * @param	fileName is the name of the file to be extracted.
	 * @param	strDestLoc is the location where file will be extracted.
	 * @param 	bFirstDirLevel to indicate to get file from first directory level.
	 * @return  Sucess or Failure
	 */
	public static boolean extractFileFromZip (String zipFileName, String fileName, String strDestLoc, boolean bFirstDirLevel) {
		boolean bRetCode = Constants.bFAILURE;
		try {
			BufferedOutputStream bufOut = null;
			BufferedInputStream bufIn = null;
			ZipEntry entry;
			ZipFile zipfile = new ZipFile(zipFileName);
			Enumeration e = zipfile.entries();
			while(e.hasMoreElements()) {
				entry = (ZipEntry) e.nextElement();
				if(! entry.getName().endsWith(fileName))
					continue;
				//Get file at root (in single directory) level. This is for License in root location.
				if( bFirstDirLevel &&
						(entry.getName().indexOf('/') != entry.getName().lastIndexOf('/')))
					continue;
				bufIn = new BufferedInputStream(zipfile.getInputStream(entry));
				int count;
				byte data[] = new byte[Constants.BUFFER];
				String strOutFileName = strDestLoc == null ? entry.getName(): strDestLoc + "/" + fileName; 
				FileOutputStream fos = new FileOutputStream(strOutFileName);
				bufOut = new BufferedOutputStream(fos, Constants.BUFFER);
				while ((count = bufIn.read(data, 0, Constants.BUFFER)) != -1) {
					bufOut.write(data, 0, count);
				}
				bufOut.flush();
				bufOut.close();
				bufIn.close();
				bRetCode = Constants.bSUCCESS;
				break;
			}
		} catch(Exception e) {
			e.printStackTrace();
		}
		return bRetCode;
	}

	/**
	 * This will return the file from tgz file and store it in specified location.
	 *
	 * @param	tgzFileName is the name of tgz file from which file to be extracted.
	 * @param	fileName is the name of the file to be extracted.
	 * @param	strDestLoc is the location where file will be extracted.
	 * @param 	bFirstDirLevel to indicate to get file from first directory level.
	 * @return	Sucess or Failure
	 */
	public static boolean extractFileFromTGZ (String tgzFileName, String fileName, String strDestLoc, boolean bFirstDirLevel) {

		boolean bRetCode = Constants.bFAILURE;

		TarArchiveInputStream tarIn = null; 

		try { 

			tarIn = new TarArchiveInputStream(
						new GzipCompressorInputStream(
							new BufferedInputStream(
								new FileInputStream(tgzFileName))));
		} catch(Exception e) {
			Utility.debugPrint(Constants.DEBUG_ERROR, "Exception in unzipping tar file: " + e);
			return bRetCode;
		} 

		try {
			BufferedOutputStream bufOut = null;
			BufferedInputStream bufIn = null;
			TarArchiveEntry tarEntry = null;
			while((tarEntry = tarIn.getNextTarEntry()) != null) {
				if(! tarEntry.getName().endsWith(fileName))
					continue;
				//Get file at root (in single directory) level. This is for License in root location.
				if( bFirstDirLevel &&
						(tarEntry.getName().indexOf('/') != tarEntry.getName().lastIndexOf('/')))
					continue;
				bufIn = new BufferedInputStream (tarIn);
				int count;
				byte data[] = new byte[Constants.BUFFER];
				String strOutFileName = strDestLoc == null ? tarEntry.getName(): strDestLoc + "/" + fileName; 
				FileOutputStream fos = new FileOutputStream(strOutFileName);
				bufOut = new BufferedOutputStream(fos, Constants.BUFFER);
				while ((count = bufIn.read(data, 0, Constants.BUFFER)) != -1) {
					bufOut.write(data, 0, count);
            			}
				bufOut.flush();
				bufOut.close();
				bufIn.close();
				bRetCode = Constants.bSUCCESS;
				break;
			}
		} catch(Exception e) {
			e.printStackTrace();
		}
		return bRetCode;
	}

	/**
	 * This will return the list of files from archive.
	 *
	 * @param	zipFileName is the name of archive (e.g., .zip/tgz/tar.gz file) from which list of files with specified file extension will be returned.
	 * @param	fileExt is the file extension to be used to get list of files to be returned.
	 * @return	Returns list of files having specified extension from archive.
	 */
	public static List<String> getFiles (String zipFileName, String fileExt) {
		if (zipFileName.endsWith("." + Constants.ZIP))
			return getFilesFromZip (zipFileName, fileExt);
		else if (zipFileName.endsWith("." + Constants.TGZ) || zipFileName.endsWith("." + Constants.TAR_GZ))
			return getFilesFromTGZ (zipFileName, fileExt);
		return null;
	}
	/**
	 * This will return the list of files from zip file.
	 *
	 * @param	zipFileName is the name of zip file from which list of files with specified file extension will be returned.
	 * @param	fileExt is the file extension to be used to get list of files to be returned.
	 * @return	Returns list of files having specified extension from zip file.
	 */
	public static List<String> getFilesFromZip (String zipFileName, String fileExt) {
		List<String> files = new ArrayList<String>();
		try {
			ZipEntry entry;
			ZipFile zipfile = new ZipFile(zipFileName);
			Enumeration e = zipfile.entries();
			while(e.hasMoreElements()) {
				entry = (ZipEntry) e.nextElement();
				if(entry.getName().endsWith("." + fileExt)) {
					int iPos = entry.getName().lastIndexOf("/");
					if (iPos == 0)
					    --iPos;
					String strFileName = entry.getName().substring(iPos+1);
					files.add(strFileName);
				}
			}
		} catch(Exception e) {
			e.printStackTrace();
		}
		return (files);
	}

	/**
	 * This will return the list of files from tgz file.
	 *
	 * @param	tgzFileName is the name of tgz file from which list of files with specified file extension will be returned.
	 * @param	fileExt is the file extension to be used to get list of files to be returned.
	 * @return	Returns list of files having specified extension from tgz file.
	 */
	public static List<String> getFilesFromTGZ (String tgzFileName, String fileExt) {

		TarArchiveInputStream tarIn = null; 

		try { 

			tarIn = new TarArchiveInputStream(
						new GzipCompressorInputStream(
							new BufferedInputStream(
								new FileInputStream(tgzFileName))));
		} catch(Exception e) {
			Utility.debugPrint(Constants.DEBUG_ERROR, "Exception in unzipping tar file: " + e);
			return null;
		} 

		List<String> files = new ArrayList<String>();
		try {
			TarArchiveEntry tarEntry = null;
			while((tarEntry = tarIn.getNextTarEntry()) != null) {
				if(tarEntry.getName().endsWith("." + fileExt)) {
					int iPos = tarEntry.getName().lastIndexOf("/");
					if (iPos == 0)
						--iPos;
					String strFileName = tarEntry.getName().substring(iPos+1);
					files.add(strFileName);
				}
			}
		} catch(Exception e) {
			e.printStackTrace();
		}
		return (files);
	}

	/**
	 * This will return if NOTICE file is valid or not.
	 *
	 * @param	noticeFile is the noticew file to be verified.
	 * @return 	Returns if NOTICE file validatation successful or failure.
	 */
	public static boolean validateNotice(String noticeFile) throws Exception {

		boolean bValidNotice = Constants.bSUCCESS;

		LocalDateTime currentTime = LocalDateTime.now();

		String noticeLines[] = new String[4];
		boolean noticeLineIn[] = new boolean[4];

// ToDo: fix notice lines
		noticeLines[0] = "SystemDS";
		noticeLines[1] = "Copyright [2018-" + currentTime.getYear() + "] Graz University of Technology";
//		noticeLines[2] = "This product includes software developed at";
//		noticeLines[3] = "The Apache Software Foundation (http://www.apache.org/)";

		BufferedReader reader = new BufferedReader(new FileReader(noticeFile));
		String line = null;
		while ((line = reader.readLine()) != null) {
			line = line.trim();
			for (int i = 0; i < noticeLines.length; i++) {
				if (line.contains(noticeLines[i])) {
					noticeLineIn[i] = true;
				}
			}
		}

		for (int i = 0; i < noticeLines.length; i++) {
			if (!noticeLineIn[i]) {
				bValidNotice = Constants.bFAILURE;
			}
		}

		if(bValidNotice == Constants.bSUCCESS)
			Utility.debugPrint(Constants.DEBUG_INFO2, "Notice validation successful.");

		return bValidNotice;
	}

	/**
	 * This will validate license for JavaScript & CSS files within an archive.
	 *
	 * @param	licenseFile is the file against which contents of archive gets compared.
	 * @param	zipFileName is the name of archive from which list of JavaScript files will be returned.
	 * @return  Success or Failure code
	 */
	public static boolean validateJSCssLicense(File licenseFile, String zipFileName) throws Exception
	{
		boolean bRetCode = Constants.bSUCCESS;

		try {
			List<String> jsFiles = getFiles(zipFileName, Constants.JS);
			List<String> cssFiles = getFiles(zipFileName, Constants.CSS);
			HashMap<String, Boolean> jsCssFileHashMap = new HashMap<String, Boolean>();
			for (String jsFile : jsFiles)
				if(jsFile.compareTo("main.js") != 0)
					jsCssFileHashMap.put(jsFile, Boolean.FALSE);
			for (String cssFile : cssFiles)
				if(cssFile.compareTo("main.css") != 0)
					jsCssFileHashMap.put(cssFile, Boolean.FALSE);

			BufferedReader reader = new BufferedReader(new FileReader(licenseFile));
			String line = null;
			HashSet<String> packageValidLic = new HashSet<String>();
			while ((line = reader.readLine()) != null) {
				line = line.trim();
				//Move to beginning of individual License text
				if (line.startsWith(Constants.LIC_TEXT_DELIM))
					break;
			}

			while ((line = reader.readLine()) != null) {
				line = line.trim();

				List<String> curLicense = new ArrayList<String>();
				//Read all lines until end of individual License text
				while (!line.startsWith(Constants.LIC_TEXT_DELIM)) {
					curLicense.add(line);
					if ((line = reader.readLine()) == null)
						break;
				}

				//Verify jsFiles against current license foumd.
				Iterator<Map.Entry<String, String[][]>> itJSLicenses = hmJSLicenses.entrySet().iterator();
				while (itJSLicenses.hasNext()) {
					Map.Entry<String, String[][]> pairJSLicense = (Map.Entry<String, String[][]>) itJSLicenses.next();

					String[][] JSLicenseList = pairJSLicense.getValue();

					for (String[] license : JSLicenseList) {
						boolean bLicFirstPartFound = false;

						for (String licLine : curLicense) {
							if (!bLicFirstPartFound && licLine.startsWith(license[0]))
								bLicFirstPartFound = true;

							if (bLicFirstPartFound && licLine.contains(license[1])) {
								license[2] = "true";
								break;
							}
						}
					}
				}
			}

			//Validate all js/css files against license found in LICENSE file.
			Iterator<Map.Entry<String, Boolean>> itJSCssFiles = jsCssFileHashMap.entrySet().iterator();
			while (itJSCssFiles.hasNext()) {
				Map.Entry<String, Boolean> pairJSCSSFile = (Map.Entry<String, Boolean>) itJSCssFiles.next();

				String[][] jsFileLicList = hmJSLicenses.get(pairJSCSSFile.getKey());
				if(jsFileLicList == null) {
					Utility.debugPrint(Constants.DEBUG_WARNING, "JS/CSS license does not exist for file " + pairJSCSSFile.getKey());
					bRetCode = Constants.bFAILURE;
					continue;
				}

				boolean bValidLic = true;
				for (String[] jsFileLic : jsFileLicList) {
					if (jsFileLic[2].compareTo("true") != 0) {
						bValidLic = false;
						break;
					}
				}

				if (bValidLic) {
					jsCssFileHashMap.put(pairJSCSSFile.getKey(), Boolean.TRUE);
					Utility.debugPrint(Constants.DEBUG_INFO3, "JS/CSS license exists for file " + pairJSCSSFile.getKey());
				}
				else {
					Utility.debugPrint(Constants.DEBUG_WARNING, "JS/CSS license does not exist for file " + pairJSCSSFile.getKey());
					bRetCode = Constants.bFAILURE;
				}
			}

			if (bRetCode == Constants.bSUCCESS)
				Utility.debugPrint(Constants.DEBUG_INFO2, "JS/CSS license validation successful.");

		} catch (Exception e) {
			System.out.println(e);
			e.printStackTrace();
		}
		return bRetCode;
	}

	/**
	 * This is main() program.
	 *
	 * @param	args is list of arguments
	 * @return
	 */
	public static void  main(String [] args) {
		ValidateLicAndNotice valLic;
		if (args.length > 0)
			valLic = new ValidateLicAndNotice(args[0]);
		else
			valLic  = new ValidateLicAndNotice();

		try { 
			int retCode = valLic.validate();

			Utility.debugPrint(Constants.DEBUG_INFO, "Return code = " + retCode);
		}
		catch (Exception e) {
			Utility.debugPrint(Constants.DEBUG_ERROR, "Error while validating license in archive." + e);
		}
	}

}
