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

package org.apache.sysds.runtime.privacy;

import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.BiFunction;

import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;

/**
 * Class counting the checked privacy constraints and the loaded privacy constraints. 
 */
public class CheckedConstraintsLog {
	private static final Map<PrivacyLevel,LongAdder> loadedConstraintsTotal = new EnumMap<>(PrivacyLevel.class);
	static {
		for ( PrivacyLevel level : PrivacyLevel.values() )
			loadedConstraintsTotal.put(level, new LongAdder());
	}
	private static final Map<PrivacyLevel,LongAdder> checkedConstraintsTotal = new EnumMap<>(PrivacyLevel.class);
	private static final BiFunction<LongAdder, LongAdder, LongAdder> mergeLongAdders = (v1, v2) -> {
		v1.add(v2.longValue() );
		return v1;
	};

	/**
	 * Adds checkedConstraints to the checked constraints total. 
	 * @param checkedConstraints constraints checked by federated worker
	 */
	public static void addCheckedConstraints(Map<PrivacyLevel,LongAdder> checkedConstraints){
		if ( checkedConstraints != null){
			checkedConstraints.forEach( 
			(key,value) -> checkedConstraintsTotal.merge( key, value, mergeLongAdders) );
		}
	}

	/**
	 * Add an occurence of the given privacy level to the loaded constraints log total. 
	 * @param level privacy level from loaded privacy constraint
	 */
	public static void addLoadedConstraint(PrivacyLevel level){
		if (level != null)
			loadedConstraintsTotal.get(level).increment();
	}

	/**
	 * Remove all elements from checked constraints log and loaded constraints log.
	 */
	public static void reset(){
		checkedConstraintsTotal.clear();
		loadedConstraintsTotal.replaceAll((k,v)->new LongAdder());
	}

	public static Map<PrivacyLevel,LongAdder> getCheckedConstraints(){
		return checkedConstraintsTotal;
	}

	public static Map<PrivacyLevel, LongAdder> getLoadedConstraints(){
		return loadedConstraintsTotal;
	}

	/**
	 * Get string representing all contents of the checked constraints log.
	 * @return string representation of checked constraints log.
	 */
	public static String display(){
		StringBuilder sb = new StringBuilder();
		sb.append("Checked Privacy Constraints:\n");
		checkedConstraintsTotal.forEach((k,v)->sb.append("\t" + k + ": " + v + "\n"));
		sb.append("Loaded Privacy Constraints:\n");
		loadedConstraintsTotal.forEach((k,v)->sb.append("\t" + k + ": " + v + "\n"));
		return sb.toString();
	}
}
