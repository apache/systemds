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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.parser.DataExpression;

import java.io.Serializable;

public class FileFormatPropertiesLIBSVM extends FileFormatProperties implements Serializable {
	protected static final Log LOG = LogFactory.getLog(FileFormatPropertiesLIBSVM.class.getName());
	private static final long serialVersionUID = -2870393360885401604L;

	private String delim;
	private String indexDelim;
	private boolean sparse;

	public FileFormatPropertiesLIBSVM() {
		// get the default values for LIBSVM properties from the language layer
		this.delim = DataExpression.DEFAULT_DELIM_DELIMITER;
		this.indexDelim = DataExpression.DEFAULT_LIBSVM_INDEX_DELIM;

		if(LOG.isDebugEnabled())
			LOG.debug("FileFormatPropertiesLIBSVM: " + this.toString());
	}

	public FileFormatPropertiesLIBSVM(String delim, String indexDelim) {
		this();
		this.delim = delim;
		this.indexDelim = indexDelim;
		if(LOG.isDebugEnabled())
			LOG.debug("FileFormatPropertiesLIBSVM full settings: " + this.toString());
	}

	public FileFormatPropertiesLIBSVM(String delim, String indexDelim, boolean sparse) {
		this();
		this.delim = delim;
		this.indexDelim = indexDelim;
		this.sparse = sparse;
		if(LOG.isDebugEnabled())
			LOG.debug("FileFormatPropertiesLIBSVM full settings: " + this.toString());
	}

	public String getDelim() {
		return delim;
	}

	public void setDelim(String delim) {
		this.delim = delim;
	}

	public String getIndexDelim() {
		return indexDelim;
	}

	public void setIndexDelim(String indexDelim) {
		this.indexDelim = indexDelim;
	}

	public boolean isSparse() {
		return sparse;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(" delim " + delim);
		sb.append(" indexDelim " + indexDelim);
		sb.append(" sparse " + sparse);
		return sb.toString();
	}
}
