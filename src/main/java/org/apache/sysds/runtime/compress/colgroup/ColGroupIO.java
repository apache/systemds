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
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.ColGroupType;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;

/** IO for ColGroups, it enables read and write ColGroups */
public interface ColGroupIO {

	static final Log LOG = LogFactory.getLog(ColGroupIO.class.getName());

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

		// Allocate that amount into an ArrayList
		final List<AColGroup> _colGroups = new ArrayList<>(nColGroups);

		// Read each ColGroup one at a time.
		for(int i = 0; i < nColGroups; i++)
			_colGroups.add(readColGroup(in, nRows));

		return _colGroups;
	}

	/**
	 * Writes the ColGroups out to the DataOutput.
	 * 
	 * @param out       The DataOutput the ColGroups are written to
	 * @param colGroups List of the ColGroups to write to file.
	 * @throws IOException Throws IO Exception if the out refuses to write.
	 */
	public static void writeGroups(DataOutput out, Collection<AColGroup> colGroups) throws IOException {
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
		Set<IDictionary> dicts = new HashSet<>();
		for(AColGroup grp : colGroups){
			if(grp instanceof ADictBasedColGroup){
				IDictionary dict = ((ADictBasedColGroup)grp).getDictionary();
				if(dicts.contains(dict))
					ret -= dict.getExactSizeOnDisk();
				dicts.add(dict);
			}
			ret += grp.getExactSizeOnDisk();
		}

		return ret;
	}

	public static AColGroup readColGroup(DataInput in, int nRows) throws IOException {
		final ColGroupType ctype = ColGroupType.values()[in.readByte()];
		switch(ctype) {
			case DDC:
				return ColGroupDDC.read(in);
			case DDCFOR:
				return ColGroupDDCFOR.read(in);
			case OLE:
				return ColGroupOLE.read(in, nRows);
			case RLE:
				return ColGroupRLE.read(in, nRows);
			case CONST:
				return ColGroupConst.read(in);
			case EMPTY:
				return ColGroupEmpty.read(in);
			case UNCOMPRESSED:
				return ColGroupUncompressed.read(in);
			case SDC:
				return ColGroupSDC.read(in, nRows);
			case SDCSingle:
				return ColGroupSDCSingle.read(in, nRows);
			case SDCSingleZeros:
				return ColGroupSDCSingleZeros.read(in, nRows);
			case SDCZeros:
				return ColGroupSDCZeros.read(in, nRows);
			case SDCFOR:
				return ColGroupSDCFOR.read(in, nRows);
			case LinearFunctional:
				return ColGroupLinearFunctional.read(in, nRows);
			default:
				throw new DMLRuntimeException("Unsupported ColGroup Type used: " + ctype);
		}
	}

	public static double[] readDoubleArray(int length, DataInput in) throws IOException {
		double[] ret = new double[length];
		for(int i = 0; i < length; i++)
			ret[i] = in.readDouble();
		return ret;
	}
}
