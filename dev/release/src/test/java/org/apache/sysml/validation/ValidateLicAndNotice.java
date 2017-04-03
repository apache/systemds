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

package org.apache.sysml.validation;

import java.io.BufferedReader;
import java.io.BufferedOutputStream;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
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
	//Return codes
	public static final int SUCCESS = 0;
	public static final int NO_ZIP_TGZ = 1;			// 0000 0000 0000 0001
	public static final int FILE_NOT_IN_LIC = 2; 	// 0000 0000 0000 0010
	public static final int FILE_NOT_IN_ZIP = 4; 	// 0000 0000 0000 0100
	public static final int FAILED_TO_EXTRACT = 8; 	// 0000 0000 0000 1000
	public static final int LIC_NOT_EXIST = 16;		// 0000 0000 0001 0000
	public static final int FAILURE = 0xFFFF;

	public static final boolean bSUCCESS = true;
	public static final boolean bFAILURE = false;


	//DEBUG PRINT CODE
	public static final int DEBUG_PRINT_LEVEL = 3;

	public static final int DEBUG_ERROR = 0;
	public static final int DEBUG_WARNING = 1;
	public static final int DEBUG_INFO = 2;
	public static final int DEBUG_INFO2 = 3;
	public static final int DEBUG_CODE = 4;


	static final int BUFFER = 2048;

	//String constants
	public static final String SYSTEMML_NAME = "SystemML";
	public static final String SYSTEMML_PACKAGE = "org/apache/sysml";

	public static final String ZIP = "zip";
	public static final String TGZ = "tgz";
	public static final String LICENSE = "LICENSE";
	public static final String JAR = "jar";
	public static final String DLL = "dll";
	public static final String EXP = "exp";
	public static final String LIB = "lib";
	public static final String PDB = "pdb";
	public static final String EXE = "exe";
	public static final String CLASS = "class";

	public static String[] fileTypes = {JAR, DLL, EXP, LIB, PDB, EXE};

	// Zip Distribution directory.
	private String strDistroDir =  "../../../target/release/incubator-systemml/target/";

	public static final ArrayList<String[]> packageLicenses = new ArrayList<String[]> ();
	static {
		String[] strA1 = {"org/antlr", "ANTLR 4 Runtime (http://www.antlr.org/antlr4-runtime) org.antlr:antlr4-runtime:4.5.3"};
		String[] strA2 = {"org/apache/wink/json4j","Apache Wink :: JSON4J (http://www.apache.org/wink/wink-json4j/) org.apache.wink:wink-json4j:1.4"};

		packageLicenses.add(strA1);
		packageLicenses.add(strA2);
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
	 * This will validate all zip and tgz from distribution location.
	 *
	 * @return Returns the output code
	 */
	public int validate() throws Exception {

		int retCode = SUCCESS, retCodeForAllFileTypes = SUCCESS, retCodeAll = SUCCESS;

		File distroRoot = new File( getDistroDir());
		File libDirectory = distroRoot;
		if (!libDirectory.exists()) {
			debugPrint(DEBUG_ERROR, "Distribution folder '" + libDirectory.getAbsoluteFile().toString() + "' does not exist.");
			return NO_ZIP_TGZ;
		}

		File outTempDir = File.createTempFile("outTemp", "");
		outTempDir.delete();
		outTempDir.mkdir();

		List<String> zips = getZipsInDistro(libDirectory);
		if(zips.size() == 0) {
			debugPrint(DEBUG_ERROR, "Can't find zip/tgz files in folder: " + libDirectory.getAbsoluteFile().toString());
			return NO_ZIP_TGZ;
		}

		for (String zipFile: zips)
		{
			retCodeForAllFileTypes = SUCCESS;
			debugPrint(DEBUG_INFO, "======================================================================================");
			debugPrint(DEBUG_INFO, "Validating zip file : " + zipFile + " ...");

			for (String fileType: fileTypes) {
				retCode = SUCCESS;

				List<String> filesAll = null;
				// Extract license only at first time in all filetypes validation for a given zip.
				if(fileType == JAR)
					if (!ValidateLicAndNotice.extractFile(libDirectory + "/" + zipFile, LICENSE, outTempDir.getAbsolutePath(), true))
						return FAILED_TO_EXTRACT;

				filesAll = getFiles(libDirectory + "/" + zipFile, fileType);

				File licenseFile = new File(outTempDir, LICENSE);
				List<String> files = new ArrayList<String>();
				List<String> fileSysml = new ArrayList<String>();
				for (String file : filesAll) {
					int sysmlLen = SYSTEMML_NAME.length();
					String strBegPart = file.substring(0, sysmlLen);
					if (strBegPart.compareToIgnoreCase(SYSTEMML_NAME) != 0)
						files.add(file);
					else
						fileSysml.add(file);
				}

				// Validate shaded jar only one time.
				if(fileType == JAR)
					for (String file : fileSysml)
						retCode += ValidateLicAndNotice.validateShadedLic(libDirectory + "/" + zipFile, file, outTempDir.getAbsolutePath());

				List<String> bad2 = getLICENSEFilesNotInList(licenseFile, files, fileType);
				if (bad2.size() > 0) {
					System.err.println("Files in LICENSE but not in Distribution: " + bad2);
					retCode += FILE_NOT_IN_ZIP;
				}

				List<String> bad1 = getFilesNotInLICENSE(licenseFile, files, fileType);
				if (bad1.size() > 0) {
					System.err.println("Files in distribution but not in LICENSE: " + bad1);
					retCode += FILE_NOT_IN_LIC;
				}

				if (retCode > SUCCESS) {
					debugPrint(DEBUG_ERROR, "License validation of file types " + fileType + " failed for zip file " + zipFile + " with error code " + retCode + ", please validate file manually.");
					retCodeForAllFileTypes = FAILURE;
				}
			}
			if(retCodeForAllFileTypes == SUCCESS)
				debugPrint(DEBUG_INFO, "Validation of zip file : " + zipFile + " completed successfully.");

			retCodeAll = retCode != SUCCESS?FAILURE:retCodeAll;
		}
		debugPrint(DEBUG_INFO, "======================================================================================");

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
			return FAILED_TO_EXTRACT;
		if(!ValidateLicAndNotice.extractFile(outTempDir2.getAbsolutePath()+"/"+file, LICENSE, outTempDir2.getAbsolutePath(), true))
			return FAILED_TO_EXTRACT;

		HashMap<String, Boolean> hashMapPackages = getPackagesFromZip(outTempDir2.getAbsolutePath() + "/" + file);
		for (String packageName: hashMapPackages.keySet())
			debugPrint(DEBUG_CODE, "Package: " + packageName + " Licensed: " + hashMapPackages.get(packageName));

		int iRetCode = ValidateLicAndNotice.validatePackages(outTempDir2.getAbsolutePath()+"/"+LICENSE, hashMapPackages);

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
		int iRetCode = SUCCESS;
		BufferedReader reader = new BufferedReader(new FileReader(licenseFile));
		String line = null;
		HashSet <String> packageValidLic = new HashSet<String>();
		while ((line = reader.readLine()) != null) {
			line = line.trim();
			for(int i=0; i <packageLicenses.size(); ++i) {
				if (line.contains(packageLicenses.get(i)[1])) {
					packageValidLic.add(packageLicenses.get(i)[0]);
					debugPrint(DEBUG_INFO2, "License for package " + packageLicenses.get(i)[0] + " exists.");
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
				debugPrint(DEBUG_WARNING, "Could not validate license for package " + pairPackage.getKey() + ", please validate manually.");
				iRetCode = LIC_NOT_EXIST;
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
				if(! entry.getName().startsWith(SYSTEMML_PACKAGE) &&
				     entry.getName().endsWith("." + CLASS)) {
					int iPos = entry.getName().lastIndexOf("/");
					if (iPos > 0) {
						String strPackageName = entry.getName().substring(0, iPos);
						packages.put(strPackageName, Boolean.FALSE);
						debugPrint(DEBUG_CODE, "Package found : " + strPackageName);
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
	 * This will return the list of files in licsense files but not in list of files coming from zip/tgz file.
	 *
	 * @param	licenseFile is the file against which contents of zip/tgz file gets compared.
	 * @param 	files	are the list of files coming from zip/tgz file.
	 * @param 	fileExt	is the extention of file to validate (e.g. "jar")
	 * @return 	Returns the list of files in License file but not in zip/tgz file.
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
	 * This will return the list of files in licsense files with specified file extention.
	 *
	 * @param	licenseFile is the file against which contents of zip/tgz file gets compared.
	 * @param 	fileExt	is the extention of file to validate (e.g. "jar")
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
	* This will return the list of files coming from zip/tgz file but not in the licsense file.
	 *
	 * @param	licenseFile is the file against which contents of zip/tgz file gets compared.
	 * @param 	files	are the list of files coming from zip/tgz file.
	 * @param 	fileExt	is the extention of file to validate (e.g. "jar")
	 * @return 	Returns the list of files in zip/tgz file but not in License file.
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
	 * This will return the list of zip/tgz files from a directory.
	 *
	 * @param	directory is the location from where list of zip/tgz will be returned.
	 * @return 	Returns the list of zip/tgz files from a directory.
	 */
	private List<String> getZipsInDistro(File directory) {
		List<String> zips = new ArrayList<String>();
		for (String fileName : directory.list())
			if ((fileName.endsWith("." + ZIP)) || (fileName.endsWith("." + TGZ)))
				zips.add(fileName);
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
	 * This will return the file from zip/tgz file and store it in specified location.
	 *
	 * @param	zipFileName is the name of zip/tgz file from which file to be extracted.
	 * @param	fileName is the name of the file to be extracted.
	 * @param	strDestLoc is the location where file will be extracted.
	 * @param 	bFirstDirLevel to indicate to get file from first directory level.
	 * @return  Success or Failure
	 */
	public static boolean extractFile(String zipFileName, String fileName, String strDestLoc, boolean bFirstDirLevel) {
		debugPrint(DEBUG_CODE, "Extracting " + fileName + " from jar/zip/tgz file " + zipFileName);
		if (zipFileName.endsWith("." + ZIP) || zipFileName.endsWith("." + JAR))
			return extractFileFromZip(zipFileName, fileName, strDestLoc, bFirstDirLevel);
		else if (zipFileName.endsWith("." + TGZ))
			return extractFileFromTGZ(zipFileName, fileName, strDestLoc, bFirstDirLevel);
		return bFAILURE;
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
		boolean bRetCode = bFAILURE;
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
				byte data[] = new byte[BUFFER];
				String strOutFileName = strDestLoc == null ? entry.getName(): strDestLoc + "/" + fileName; 
				FileOutputStream fos = new FileOutputStream(strOutFileName);
				bufOut = new BufferedOutputStream(fos, BUFFER);
				while ((count = bufIn.read(data, 0, BUFFER)) != -1) {
					bufOut.write(data, 0, count);
				}
				bufOut.flush();
				bufOut.close();
				bufIn.close();
				bRetCode = bSUCCESS;
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

		boolean bRetCode = bFAILURE;

		TarArchiveInputStream tarIn = null; 

		try { 

			tarIn = new TarArchiveInputStream(
						new GzipCompressorInputStream(
							new BufferedInputStream(
								new FileInputStream(tgzFileName))));
		} catch(Exception e) {
			debugPrint(DEBUG_ERROR, "Exception in unzipping tar file: " + e);
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
				byte data[] = new byte[BUFFER];
				String strOutFileName = strDestLoc == null ? tarEntry.getName(): strDestLoc + "/" + fileName; 
				FileOutputStream fos = new FileOutputStream(strOutFileName);
				bufOut = new BufferedOutputStream(fos, BUFFER);
				while ((count = bufIn.read(data, 0, BUFFER)) != -1) {
					bufOut.write(data, 0, count);
            			}
				bufOut.flush();
				bufOut.close();
				bufIn.close();
				bRetCode = bSUCCESS;
				break;
			}
		} catch(Exception e) {
			e.printStackTrace();
		}
		return bRetCode;
	}

	/**
	 * This will return the list of files from zip/tgz file.
	 *
	 * @param	zipFileName is the name of zip file from which list of files with specified file extension will be returned.
	 * @param	fileExt is the file extension to be used to get list of files to be returned.
	 * @return	Returns list of files having specified extention from zip file .
	 */
	public static List<String> getFiles (String zipFileName, String fileExt) {
		if (zipFileName.endsWith("." + ZIP))
			return getFilesFromZip (zipFileName, fileExt);
		else if (zipFileName.endsWith("." + TGZ))
			return getFilesFromTGZ (zipFileName, fileExt);
		return null;
	}
	/**
	 * This will return the list of files from zip file.
	 *
	 * @param	zipFileName is the name of zip file from which list of files with specified file extension will be returned.
	 * @param	fileExt is the file extension to be used to get list of files to be returned.
	 * @return	Returns list of files having specified extention from zip file .
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
	 * @return	Returns list of files having specified extention from tgz file .
	 */
	public static List<String> getFilesFromTGZ (String tgzFileName, String fileExt) {

		TarArchiveInputStream tarIn = null; 

		try { 

			tarIn = new TarArchiveInputStream(
						new GzipCompressorInputStream(
							new BufferedInputStream(
								new FileInputStream(tgzFileName))));
		} catch(Exception e) {
			debugPrint(DEBUG_ERROR, "Exception in unzipping tar file: " + e);
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

			debugPrint(DEBUG_INFO, "Return code = " + retCode);
		}
		catch (Exception e) {
			debugPrint(DEBUG_ERROR, "Error while validating license in zip/tgz file." + e);
		}
	}

	/**
	 * This will be used to output on console in consistent and controlled based on DEBUG flag.
	 *
	 * @param	debugLevel is the debuglevel message user wants to prrint message.
	 * @param 	message is the message to be displayed.
	 * @return
	 */
	public static void debugPrint(int debugLevel, String message) {
		String displayMessage = "";
		switch (debugLevel) {
			case DEBUG_ERROR:
				displayMessage = "ERROR: " + message;
				break;
			case DEBUG_WARNING:
				displayMessage = "WARNING: " + message;
				break;
			case DEBUG_INFO:
				displayMessage = "INFO: " + message;
				break;
			case DEBUG_INFO2:
				displayMessage = "INFO2: " + message;
				break;
			case DEBUG_CODE:
				displayMessage = "DEBUG: " + message;
				break;
			default:
				break;
		}

		if(debugLevel <= DEBUG_PRINT_LEVEL)
			System.out.println(displayMessage);
	}

}
