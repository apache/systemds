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


/**
 * This class includes constants used across validation programs.
 */
public class Constants
{
	//Return codes
	public static final int SUCCESS = 0;
	public static final int NO_ZIP_TGZ = 1;			// 0000 0000 0000 0001
	public static final int FILE_NOT_IN_LIC = 2; 	// 0000 0000 0000 0010
	public static final int FILE_NOT_IN_ZIP = 4; 	// 0000 0000 0000 0100
	public static final int FAILED_TO_EXTRACT = 8; 	// 0000 0000 0000 1000
	public static final int LIC_NOT_EXIST = 16;		// 0000 0000 0001 0000
	public static final int JS_CSS_LIC_NOT_EXIST = 32;	// 0000 0000 0010 0000
	public static final int INVALID_NOTICE = 64;	// 0000 0000 0100 0000
	public static final int FAILED_TO_CREATE_DIR = 128;	// 0000 0000 1000 0000
	public static final int FAILURE = 0xFFFF;

	public static final boolean bSUCCESS = true;
	public static final boolean bFAILURE = false;


	//DEBUG PRINT CODE
	public static final int DEBUG_PRINT_LEVEL = 3;

	public static final int DEBUG_ERROR = 0;
	public static final int DEBUG_WARNING = 1;
	public static final int DEBUG_INFO = 2;
	public static final int DEBUG_INFO2 = 3;
	public static final int DEBUG_INFO3 = 4;
	public static final int DEBUG_CODE = 5;

	static final int BUFFER = 2048;

	//String constants
	public static final String SYSTEMDS_NAME = "SystemDS";
	public static final String SYSTEMDS_PACKAGE = "org/tugraz/sysds";

	public static final String ZIP = "zip";
	public static final String TGZ = "tgz";
	public static final String TAR_GZ = "tar.gz";
	public static final String LICENSE = "LICENSE";
	public static final String NOTICE = "NOTICE";
	public static final String JAR = "jar";
	public static final String DLL = "dll";
	public static final String EXP = "exp";
	public static final String LIB = "lib";
	public static final String PDB = "pdb";
	public static final String EXE = "exe";
	public static final String CLASS = "class";
	public static final String JS = "js";
	public static final String CSS = "css";
	public static final String LIC_TEXT_DELIM = "=====";

}
