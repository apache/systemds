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

package com.ibm.bi.dml.runtime.matrix.sort;

import org.apache.hadoop.io.DoubleWritable;

public class IndexSortComparableDesc extends IndexSortComparable
{
	
	@Override
	public int compareTo(Object o) 
	{
		//descending order (note: we cannot just inverted the ascending order)
		if( o instanceof DoubleWritable ) {
			int tmp = _dval.compareTo((DoubleWritable) o);
			return (( tmp!=0 ) ? -1*tmp : tmp); //prevent -0
		}
		//compare double value and index (e.g., for stable sort)
		else if( o instanceof IndexSortComparable) {
			IndexSortComparable that = (IndexSortComparable)o;
			int tmp = _dval.compareTo(that._dval);
			tmp = (( tmp!=0 ) ? -1*tmp : tmp); //prevent -0
			if( tmp==0 ) //secondary sort
				tmp = _lval.compareTo(that._lval);
			return tmp;
		}	
		else {
			throw new RuntimeException("Unsupported comparison involving class: "+o.getClass().getName());
		}
		
	}
}
