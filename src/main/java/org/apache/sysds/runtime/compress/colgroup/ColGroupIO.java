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

package org.apache.sysds.runtime.compress.colgroup;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.ColGroupType;

/**
 * This has the IO responsibility of ColGroups, such that it enables to read and write ColGroups to and from a DataInput
 * and DataOutput
 */
public class ColGroupIO {

	protected static final Log LOG = LogFactory.getLog(ColGroupIO.class.getName());

	/**
	 * Read groups from a file. Note that the information about how many should be in the file already.
	 * 
	 * @param in    The Data input object to read from.
	 * @param nRows The number of rows in the read groups.
	 * @return Return a List containing the ColGroups from the DataInput.
	 * @throws IOException Throws IO Exception if the in refuses to read data.
	 */
	public static List<AColGroup> readGroups(DataInput in, int nRows) throws IOException {

		// Read in how many colGroups there are
		final int nColGroups = in.readInt();
		if(LOG.isDebugEnabled())
			LOG.debug("reading " + nColGroups + " ColGroups");
		
		// Allocate that amount into an ArrayList
		final List<AColGroup> _colGroups = new ArrayList<>(nColGroups);

		// Read each ColGroup one at a time.
		for(int i = 0; i < nColGroups; i++) {
			ColGroupType ctype = ColGroupType.values()[in.readByte()];
			if(LOG.isTraceEnabled())
				LOG.trace("Reading in : " + ctype);
			final AColGroup grp = constructColGroup(ctype, nRows);
			grp.readFields(in);
			_colGroups.add(grp);
		}

		return _colGroups;
	}

	/**
	 * Writes the ColGroups out to the DataOutput.
	 * 
	 * @param out       The DataOutput the ColGroups are written to
	 * @param colGroups List of the ColGroups to write to file.
	 * @throws IOException Throws IO Exception if the out refuses to write.
	 */
	public static void writeGroups(DataOutput out, List<AColGroup> colGroups) throws IOException {
		// Write out how many ColGroups to save.
		out.writeInt(colGroups.size());
		for(AColGroup grp : colGroups)
			grp.write(out);
	}

	/**
	 * Get the size on disk for the given list of column groups
	 * 
	 * @param colGroups A List of column groups to see the disk space required for.
	 * @return The exact disk size required for writing the compressed matrix.
	 */
	public static long getExactSizeOnDisk(List<AColGroup> colGroups) {
		long ret = 4; // int for number of colGroups.
		for(AColGroup grp : colGroups) 
			ret += grp.getExactSizeOnDisk();
		return ret;
	}

	private static AColGroup constructColGroup(ColGroupType ctype, int nRows){
		switch(ctype) {
			case UNCOMPRESSED:
				return new ColGroupUncompressed();
			case OLE:
				return new ColGroupOLE(nRows);
			case RLE:
				return new ColGroupRLE(nRows);
			case DDC:
				return new ColGroupDDC(nRows);
			case CONST:
				return new ColGroupConst();
			case EMPTY:
				return new ColGroupEmpty();
			case SDC:
				return new ColGroupSDC(nRows);
			case SDCSingle:
				return new ColGroupSDCSingle(nRows);
			case SDCSingleZeros:
				return new ColGroupSDCSingleZeros(nRows);
			case SDCZeros:
				return new ColGroupSDCZeros(nRows);
			case PFOR:
				return new ColGroupPFOR(nRows);
			default:
				throw new DMLRuntimeException("Unsupported ColGroup Type used:  " + ctype);
		}
	}
}
