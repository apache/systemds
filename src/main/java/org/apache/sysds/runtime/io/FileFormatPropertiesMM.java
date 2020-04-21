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

package org.apache.sysds.runtime.io;

import java.io.Serializable;
import java.util.StringTokenizer;

import org.apache.sysds.runtime.DMLRuntimeException;

public class FileFormatPropertiesMM extends FileFormatProperties implements Serializable
{
	private static final long serialVersionUID = -2870393360885401604L;
	
	public enum MMFormat {
		COORDINATE,
		ARRAY;
		@Override
		public String toString() {
			return this.name().toLowerCase();
		}
	}
	
	public enum MMField {
		REAL,
		INTEGER,
		COMPLEX,
		PATTERN;
		@Override
		public String toString() {
			return this.name().toLowerCase();
		}
	}
	
	public enum MMSymmetry {
		GENERAL,
		SYMMETRIC,
		SKEW_SYMMETRIC;
		@Override
		public String toString() {
			return this.name().toLowerCase().replaceAll("_", "-");
		}
	}
	
	private final MMFormat _fmt;
	private final MMField _field;
	private final MMSymmetry _symmetry;
	
	public FileFormatPropertiesMM() {
		// get the default values for MM properties
		this(MMFormat.COORDINATE, MMField.REAL, MMSymmetry.GENERAL);
	}
	
	public FileFormatPropertiesMM(MMFormat fmt, MMField field, MMSymmetry symmetry) {
		_fmt = fmt;
		_field = field;
		_symmetry = symmetry;
		
		//check valid combination
		if( _field == MMField.PATTERN && (_fmt == MMFormat.ARRAY || _symmetry == MMSymmetry.SKEW_SYMMETRIC) ) {
			throw new DMLRuntimeException("MatrixMarket: Invalid combination: "
				+ _fmt.toString() + " " + _field.toString() + " " + _symmetry.toString() +".");
		}
	}
	
	public MMFormat getFormat() {
		return _fmt;
	}
	
	public MMField getField() {
		return _field;
	}
	
	public MMSymmetry getSymmetry() {
		return _symmetry;
	}
	
	public boolean isIntField() {
		return _field == MMField.INTEGER;
	}
	
	public boolean isPatternField() {
		return _field == MMField.PATTERN;
	}
	
	public boolean isSymmetric() {
		return _symmetry == MMSymmetry.SYMMETRIC
			|| _symmetry == MMSymmetry.SKEW_SYMMETRIC;
	}

	public static FileFormatPropertiesMM parse(String header) {
		//example: %%MatrixMarket matrix coordinate real general
		//(note: we use a string tokenizer because the individual
		//components can be separated by an arbitrary number of spaces)
		
		StringTokenizer st = new StringTokenizer(header, " ");
		
		//check basic structure and matrix object
		int numTokens = st.countTokens();
		if( numTokens != 5 )
			throw new DMLRuntimeException("MatrixMarket: Incorrect number of header tokens: "+numTokens+" (expeced: 5).");
		String type = st.nextToken();
		if( !type.equals("%%MatrixMarket") )
			throw new DMLRuntimeException("MatrixMarket: Incorrect header start: "+type+" (expected: %%MatrixMarket).");
		String object = st.nextToken();
		if( !object.equals("matrix") )
			throw new DMLRuntimeException("MatrixMarket: Incorrect object: "+object+" (expected: matrix).");
		
		//check format, field, and 
		String format = st.nextToken();
		MMFormat fmt = null;
		switch( format ) {
			//case "array": fmt = MMFormat.ARRAY; break;
			case "coordinate": fmt = MMFormat.COORDINATE; break;
			default: throw new DMLRuntimeException("MatrixMarket: "
				+ "Incorrect format: "+format+" (expected coordinate).");
		}
		String field = st.nextToken();
		MMField f = null;
		switch( field ) {
			case "real": f = MMField.REAL; break;
			case "integer": f = MMField.INTEGER; break;
			case "pattern": f = MMField.PATTERN; break;
			//note: complex not supported
			default: throw new DMLRuntimeException("MatrixMarket: "
				+ "Incorrect field: "+field+" (expected real | integer | pattern).");
		}
		String symmetry = st.nextToken();
		MMSymmetry s = null;
		switch( symmetry ) {
			case "general": s = MMSymmetry.GENERAL; break;
			case "symmetric": s = MMSymmetry.SYMMETRIC; break;
			//case "skew-symmetric": s = MMSymmetry.SKEW_SYMMETRIC; break; //not support in R
			//note: Hermitian not supported
			default: throw new DMLRuntimeException("MatrixMarket: "
				+ "Incorrect symmetry: "+symmetry+" (expected general | symmetric).");
		}
		
		//construct file properties and check valid combination
		return new FileFormatPropertiesMM(fmt, f, s);
	}
}
