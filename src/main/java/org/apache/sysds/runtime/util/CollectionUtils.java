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

package org.apache.sysds.runtime.util;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Set;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public class CollectionUtils {

	@SafeVarargs
	public static <T> List<T> asList(List<T>... inputs) {
		List<T> ret = new ArrayList<>();
		for( List<T> list : inputs )
			ret.addAll(list);
		return ret;
	}
	
	@SafeVarargs
	public static <T> ArrayList<T> asArrayList(T... inputs) {
		ArrayList<T> ret = new ArrayList<>();
		for( T list : inputs )
			ret.add(list);
		return ret;
	}
	
	@SafeVarargs
	public static <T> Set<T> asSet(T... inputs) {
		Set<T> ret = new HashSet<>();
		for( T element : inputs )
			ret.add(element);
		return ret;
	}
	
	@SafeVarargs
	public static <T> Set<T> asSet(T[]... inputs) {
		Set<T> ret = new HashSet<>();
		for( T[] input : inputs )
			for( T element : input )
				ret.add(element);
		return ret;
	}
	
	@SafeVarargs
	public static <T> Set<T> asSet(List<T>... inputs) {
		Set<T> ret = new HashSet<>();
		for( List<T> list : inputs )
			ret.addAll(list);
		return ret;
	}
	
	public static <T> Stream<T> getStream(Iterator<T> iter) {
		Iterable<T> iterable = () -> iter;
		return StreamSupport.stream(iterable.spliterator(), false);
	}

	public static <T> boolean equals(List<T> a, List<T> b) {
		//basic checks for early abort
		if( a == b ) return true; //incl both null
		if( a == null || b == null || a.size() != b.size() )
			return false;
		ListIterator<T> iter1 = a.listIterator();
		ListIterator<T> iter2 = b.listIterator();
		while( iter1.hasNext() ) //equal length
			if( !iter1.next().equals(iter2.next()) )
				return false;
		return true;
	}
	
	public static <T> boolean containsAny(Collection<T> a, Collection<T> b) {
		//build probe table for constant-time lookups (reuse hashsets)
		Collection<T> tmp1 = a.size() < b.size() ? a : b;
		Set<T> probe = (tmp1 instanceof HashSet) ?
			(Set<T>) tmp1 : new HashSet<>(tmp1);
		//probe if there is a non-empty intersection
		Collection<T> tmp2 = (a.size() < b.size() ? b : a);
		for( T item : tmp2 )
			if( probe.contains(item) )
				return true;
		return false;
	}
	
	@SuppressWarnings("unchecked")
	public static <T> List<T> unionDistinct(List<T> a, List<T> b) {
		List<T> ret = new ArrayList<>(); // in-order results
		Set<T> probe = new HashSet<>();  // constant-time probe table
		for(List<T> list : new List[] {a,b})
			for( T item : list )
				if( !probe.contains(item) ) {
					ret.add(item);
					probe.add(item);
				}
		return ret;
	}
	
	public static <T> List<T> unionAll(List<T> a, List<T> b) {
		return CollectionUtils.asList(a, b);
	}
	

	public static <T> List<T> except(List<T> a, List<T> exceptions) {
		List<T> ret = new ArrayList<>();
		Set<T> probe = new HashSet<>(exceptions);
		for( T item : a )
			if( !probe.contains(item) )
				ret.add(item);
		return ret;
	}
	
	public static <T> void addAll(Collection<T> a, T[] b) {
		for( T item : b )
			a.add(item);
	}

	public static <T> int cardinality(T a, List<T> b) {
		int count = 0;
		for(T item : b)
			count += a.equals(item) ? 1 : 0;
		return count;
	}
}
