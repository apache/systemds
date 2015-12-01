/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */


package org.apache.sysml.runtime.matrix.data;

import org.apache.hadoop.io.Writable;

public class IdenticalConverter implements Converter<Writable, Writable, Writable, Writable>
{
	
	private Pair<Writable, Writable> pair=new Pair<Writable, Writable>();
	private boolean hasValue=false;
	
	@Override
	public void convert(Writable k1, Writable v1) {
		pair.set(k1, v1);
		hasValue=true;
	}

	public boolean hasNext() {
		return hasValue;
	}


	public Pair<Writable, Writable> next() {
		if(!hasValue)
			return null;
		
		hasValue=false;
		return pair;
	}

	@Override
	public void setBlockSize(int rl, int cl) {
	}
}
