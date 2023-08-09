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

package org.apache.sysds.runtime.compress.lib;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ADictBasedColGroup;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.PlaceHolderDict;

/**
 * The job of this Lib is to separate and combine back a list of column groups from its dictionaries, and vice versa
 * combine back together dictionaries with their respective column groups
 */
public interface CLALibSeparator {
	public static final Log LOG = LogFactory.getLog(CLALibSeparator.class.getName());

	/**
	 * Split a given list of column groups into separate subparts.
	 * 
	 * @param gs the list of groups to separate.
	 * @return A split of the groups and their dictionaries.
	 */
	public static SeparatedGroups split(List<AColGroup> gs) {
		List<IDictionary> dicts = new ArrayList<>();
		List<AColGroup> indexStructures = new ArrayList<>();
		for(AColGroup g : gs) {
			if(g instanceof ADictBasedColGroup) {
				ADictBasedColGroup dg = (ADictBasedColGroup) g;
				dicts.add(dg.getDictionary());
				indexStructures.add(dg.copyAndSet(new PlaceHolderDict(dg.getNumValues())));
			}
			else {
				// add a placeholder if not a Dict based thing
				dicts.add(new PlaceHolderDict(-1));
				indexStructures.add(g);
			}
		}

		return new SeparatedGroups(dicts, indexStructures);
	}

	/**
	 * Combine a set of separated groups back together.
	 * 
	 * @param gs   groups to combine with dictionaries
	 * @param d    dictionaries to combine back into the groups.
	 * @param blen The block size.
	 * @return A combined list of columngroups.
	 */
	public static List<AColGroup> combine(List<AColGroup> gs, Map<Integer, List<IDictionary>> d, int blen) {
		int gid = 0;

		for(int i = 0; i < d.size(); i++) {
			List<IDictionary> dd = d.get(i);
			for(int j = 0; j < dd.size(); j++) {
				IDictionary ddd = dd.get(j);
				if(!(ddd instanceof PlaceHolderDict)) {

					AColGroup g = gs.get(gid);
					while(!(g instanceof ADictBasedColGroup)) {
						gid++;
						g = gs.get(gid);
					}
					ADictBasedColGroup dg = (ADictBasedColGroup) g;

					gs.set(gid, dg.copyAndSet(ddd));
				}

				gid++;
			}
		}

		return gs;
	}

	public static class SeparatedGroups {
		public final List<IDictionary> dicts;
		public final List<AColGroup> indexStructures;

		private SeparatedGroups(List<IDictionary> dicts, List<AColGroup> indexStructures) {
			this.dicts = dicts;
			this.indexStructures = indexStructures;
		}

		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder();
			sb.append(dicts);
			sb.append(indexStructures);
			return sb.toString();
		}
	}

}
