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

package org.apache.sysds.runtime.iogen;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.io.FileFormatProperties;

import java.io.Serializable;
import java.util.HashSet;

public class FileFormatPropertiesGR extends FileFormatProperties implements Serializable {
	protected static final Log LOG = LogFactory.getLog(FileFormatPropertiesGR.class.getName());
	private static final long serialVersionUID = -4447926749068752721L;

	private String delim;
	private String indexDelim;
	private HashSet<String> naStrings; // default null

	public enum GRPattern {
		Regular, Irregular;

		@Override public String toString() {
			return this.name().toLowerCase();
		}
	}

	public enum GRSymmetry {
		GENERAL, SYMMETRIC, SKEW_SYMMETRIC;

		@Override public String toString() {
			return this.name().toLowerCase().replaceAll("_", "-");
		}
	}

	private final GRPattern rowPattern;
	private final GRPattern colPattern;
	private final GRSymmetry grSymmetry;
	private int ncols;

	// CSV format
	public FileFormatPropertiesGR(GRPattern rowPattern, String delim, HashSet<String> naStrings) {
		this.delim = delim;
		this.naStrings = naStrings;
		this.rowPattern = rowPattern;
		this.colPattern = GRPattern.Regular;
		this.grSymmetry = GRSymmetry.GENERAL;
	}

	// LIBSVM format
	public FileFormatPropertiesGR(GRPattern rowPattern, String delim, String indexDelim) {
		this.delim = delim;
		this.indexDelim = indexDelim;
		this.rowPattern = rowPattern;
		this.colPattern = GRPattern.Irregular;
		this.grSymmetry = GRSymmetry.GENERAL;
	}

	// Matrix Market format
	public FileFormatPropertiesGR(GRSymmetry grSymmetry, String delim) {
		this.delim = delim;
		this.grSymmetry = grSymmetry;
		this.colPattern = GRPattern.Regular;
		this.rowPattern = GRPattern.Irregular;
	}

	public int getNcols() {
		return ncols;
	}

	public void setNcols(int ncols) {
		this.ncols = ncols;
	}

	public String getDelim() {
		return delim;
	}

	public String getIndexDelim() {
		return indexDelim;
	}

	public HashSet<String> getNaStrings() {
		return naStrings;
	}

	public GRPattern getRowPattern() {
		return rowPattern;
	}

	public GRPattern getColPattern() {
		return colPattern;
	}

	public GRSymmetry getGrSymmetry() {
		return grSymmetry;
	}
}
