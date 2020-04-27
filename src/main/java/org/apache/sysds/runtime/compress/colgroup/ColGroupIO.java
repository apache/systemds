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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.colgroup.ColGroup.ColGroupType;

/**
 * This has the IO responsibility of ColGroups, such that it enables to read and write ColGroups to and from a DataInput
 * and DataOutput
 */
public class ColGroupIO {

	/**
	 * Read groups from a file. Note that the information about how many should be in the file already.
	 * 
	 * @param in              The Data input object to read from.
	 * @param _sharedDDC1Dict Boolean flag to specify if the DCC should share dictionary.
	 * @return Return a List containing the ColGroups from the DataInput.
	 * @throws IOException Throws IO Exception if the in refuses to read data.
	 */
	public static List<ColGroup> readGroups(DataInput in, boolean _sharedDDC1Dict) throws IOException {

		// Read in how many colGroups there are
		int nColGroups = in.readInt();

		// Allocate that amount into an ArrayList
		List<ColGroup> _colGroups = new ArrayList<>(nColGroups);
		double[] sharedDict = null;

		// Read each ColGroup one at a time.
		for(int i = 0; i < nColGroups; i++) {
			ColGroupType ctype = ColGroupType.values()[in.readByte()];
			ColGroup grp = null;

			// create instance of column group
			switch(ctype) {
				case UNCOMPRESSED:
					grp = new ColGroupUncompressed();
					break;
				case OLE:
					grp = new ColGroupOLE();
					break;
				case RLE:
					grp = new ColGroupRLE();
					break;
				case DDC1:
					grp = new ColGroupDDC1();
					break;
				case DDC2:
					grp = new ColGroupDDC2();
					break;
				default:
					throw new DMLRuntimeException("Unsupported ColGroup Type used");
			}

			// Deserialize and add column group (flag for shared dictionary passed
			// and numCols evaluated in DDC1 because numCols not available yet
			grp.readFields(in, sharedDict != null);

			// use shared DDC1 dictionary if applicable
			if(_sharedDDC1Dict && grp.getNumCols() == 1 && grp instanceof ColGroupDDC1) {
				if(sharedDict == null)
					sharedDict = ((ColGroupValue) grp).getValues();
				else
					((ColGroupValue) grp).setValues(sharedDict);
			}

			_colGroups.add(grp);
		}

		return _colGroups;
	}

	/**
	 * Writes the ColGroups out to the DataOutput.
	 * 
	 * @param out             The DataOutput the ColGroups are written to
	 * @param _sharedDDC1Dict Boolean flag specifying if the DDC share dictionaries.
	 * @param _colGroups      List of the ColGroups to write to file.
	 * @throws IOException Throws IO Exception if the out refuses to write.
	 */
	public static void writeGroups(DataOutput out, boolean _sharedDDC1Dict, List<ColGroup> _colGroups)
		throws IOException
	{
		// Write out how many ColGroups we save.
		out.writeInt(_colGroups.size());

		boolean skipDict = false;
		for(ColGroup grp : _colGroups) {
			// TODO save DDC Dict sharing smarter.
			boolean shared = (grp instanceof ColGroupDDC1 && _sharedDDC1Dict && grp.getNumCols() == 1);
			out.writeByte(grp.getCompType().ordinal());
			grp.write(out, skipDict & shared); 
			skipDict |= shared;
		}
	}
}
