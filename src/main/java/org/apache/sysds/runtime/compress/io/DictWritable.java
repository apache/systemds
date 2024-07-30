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

package org.apache.sysds.runtime.compress.io;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.io.Writable;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;

public class DictWritable implements Writable, Serializable {
	private static final long serialVersionUID = 731937201435558L;
	
	public List<IDictionary> dicts;

	public DictWritable() {

	}

	protected DictWritable(List<IDictionary> dicts) {
		this.dicts = dicts;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		// the dicts can contain duplicates.
		// to avoid writing duplicates we run though once to detect them
		Set<IDictionary> ud = new HashSet<>();
		for(IDictionary d: dicts){
			if(ud.contains(d)){
				writeWithDuplicates(out);
				return;
			}
			ud.add(d);
		}

		out.writeInt(dicts.size());
		for(int i = 0; i < dicts.size(); i++)
			dicts.get(i).write(out);
	}

	private void writeWithDuplicates(DataOutput out) throws IOException {
		// indicate that we use duplicate detection
		out.writeInt(dicts.size() * -1); 
		Map<IDictionary, Integer> m = new HashMap<>();
		
		for(int i = 0; i < dicts.size(); i++){
			int id = m.getOrDefault(dicts.get(i), m.size() );
			out.writeInt(id);

			if(!m.containsKey(dicts.get(i))){
				m.put(dicts.get(i), m.size());
				dicts.get(i).write(out);
			}

		}
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		int s = in.readInt();
		if( s < 0){
			readFieldsWithDuplicates(Math.abs(s), in);
		}
		else{
			dicts = new ArrayList<>(s);
			for(int i = 0; i < s; i++)
				dicts.add(DictionaryFactory.read(in));
		}
	}

	private void readFieldsWithDuplicates(int s, DataInput in) throws IOException {

		dicts = new ArrayList<>(s);
		for(int i = 0; i < s; i++){
			int id = in.readInt();
			if(id < i)
				dicts.set(i, dicts.get(id));
			else
				dicts.add(DictionaryFactory.read(in));
		}
	}


	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("Written dictionaries:\n");
		for(IDictionary d : dicts) {
			sb.append(d);
			sb.append("\n");

		}
		return sb.toString();
	}

	public static class K implements Writable, Serializable {
		private static final long serialVersionUID = 733937201435558L;
		public int id;

		public K() {

		}

		public K(int id) {
			this.id = id;
		}

		@Override
		public void write(DataOutput out) throws IOException {
			out.writeInt(id);
		}

		@Override
		public void readFields(DataInput in) throws IOException {
			id = in.readInt();
		}

	}
}
