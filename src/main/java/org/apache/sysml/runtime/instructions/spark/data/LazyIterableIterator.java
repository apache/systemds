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

package org.apache.sysml.runtime.instructions.spark.data;

import java.util.Iterator;

/**
 * This class is a generic base class for lazy, single pass iterator classes 
 * in order to simplify the implementation of lazy iterators for mapPartitions 
 * use cases. Note [SPARK-3369], which gives the reasons for backwards compatibility
 * with regard to the iterable API despite Spark's single pass nature. 
 * 
 */
public abstract class LazyIterableIterator<T> implements Iterable<T>, Iterator<T>
{
	private Iterator<T> _iterator = null;
	private boolean _consumed = false;

	public LazyIterableIterator(Iterator<T> in) 
	{
		_iterator = in;
		_consumed = false;
	}

	/**
	 * Block computation to be overwritten by sub classes.
	 * 
	 * @param arg
	 * @return
	 * @throws Exception 
	 */
	protected abstract T computeNext(T arg) 
		throws Exception;

	//////////////////
	// iterable implementation
	
	@Override
	public Iterator<T> iterator() 
	{
		//check for consumed iterator
		if( _consumed )
			throw new RuntimeException("Invalid repeated iterator consumption.");
		
		//hand out self as iterator exactly once (note: do not hand out the input 
		//iterator since it is consumed by the self iterator implementation)
		_consumed = true;
		return this;
	}
	
	//////////////////
	// iterator implementation

	@Override
	public boolean hasNext() {
		return _iterator.hasNext();
	}
	
	@Override
	public T next() {
		try {
			return computeNext( _iterator.next() );
		}
		catch(Exception ex){
			throw new RuntimeException(ex);
		}
	}

	@Override
	public void remove() {
		throw new RuntimeException("Unsupported remove operation.");
	}
}
