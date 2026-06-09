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

package org.apache.sysds.runtime.transform;

import java.io.Serializable;

import org.apache.sysds.lops.Lop;

public class TfUtils implements Serializable
{	
	private static final long serialVersionUID = 526252850872633125L;

	protected enum ColumnTypes { 
		SCALE,
		NOMINAL,
		ORDINAL,
		DUMMYCODED;
		protected byte toID() { 
			switch(this) {
				case SCALE: return 1;
				case NOMINAL: return 2;
				case ORDINAL: return 3;
				// Ideally, dummycoded columns should be of a different type. Treating them as SCALE is incorrect, semantically.
				case DUMMYCODED: return 1; 
				default:
					throw new RuntimeException("Invalid Column Type: " + this);
			}
		}
	}
	
	//transform methods
	public enum TfMethod {
		IMPUTE, RECODE, HASH, BIN, DUMMYCODE, UDF, OMIT, WORD_EMBEDDING, BAG_OF_WORDS, RAGGED;
		@Override
		public String toString() {
			return name().toLowerCase();
		}
	}
	
	//transform meta data constants (frame-based transform)
	public static final String TXMTD_MVPREFIX = "#Meta"+Lop.DATATYPE_PREFIX+"MV";
	public static final String TXMTD_NDPREFIX = "#Meta"+Lop.DATATYPE_PREFIX+"ND";
	
	//transform meta data constants (old file-based transform)
	public static final String TXMTD_SEP         = ",";
	public static final String TXMTD_COLNAMES    = "column.names";
	public static final String TXMTD_RCD_MAP_SUFFIX      = ".map";
	public static final String TXMTD_RCD_DISTINCT_SUFFIX = ".ndistinct";
	public static final String TXMTD_BIN_FILE_SUFFIX     = ".bin";
	public static final String TXMTD_MV_FILE_SUFFIX      = ".impute";
	
	public static final String JSON_ATTRS  = "attributes";
	public static final String JSON_MTHD   = "methods";
	public static final String JSON_CONSTS = "constants";
	public static final String JSON_NBINS  = "numbins";

	public static final String EXT_SPACE = Lop.DATATYPE_PREFIX 
		+ Lop.INSTRUCTION_DELIMITOR +Lop.DATATYPE_PREFIX;
	
	private String[] _NAstrings = null;
	
	public String[] getNAStrings() { return _NAstrings; }
	
	/**
	 * Function that checks if the given string is one of NA strings.
	 * 
	 * @param NAstrings array of NA strings
	 * @param w string to check
	 * @return true if w is a NAstring
	 */
	public static boolean isNA(String[] NAstrings, String w) {
		if(NAstrings == null)
			return false;
		
		for(String na : NAstrings) {
			if(w.equals(na))
				return true;
		}
		return false;
	}
	
	public static String sanitizeSpaces(String token) {
		//due to the use of textcell (derived from matrix market), we cannot
		//simply quote the token as its parsed without support for quoting.
		//Hence, we replace with a very unlikely character sequence
		return token != null && token.contains(" ") ?
			token.replaceAll(" ", EXT_SPACE): token;
	}
	
	public static String desanitizeSpaces(String token) {
		return token != null && token.contains(EXT_SPACE) ?
			token.replaceAll(EXT_SPACE, " ") : token;
	}
}
