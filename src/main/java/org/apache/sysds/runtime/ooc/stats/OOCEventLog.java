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

package org.apache.sysds.runtime.ooc.stats;

import org.apache.sysds.api.DMLScript;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

public class OOCEventLog {
	private static final AtomicInteger _callerCtr = new AtomicInteger(0);
	private static final ConcurrentHashMap<Integer, String> _callerNames = new ConcurrentHashMap<>();
	private static final ConcurrentHashMap<String, Object> _runSettings = new ConcurrentHashMap<>();

	private static final AtomicInteger _logCtr = new AtomicInteger(0);
	private static EventType[] _eventTypes;
	private static long[] _startTimestamps;
	private static long[] _endTimestamps;
	private static int[] _callerIds;
	private static long[] _threadIds;
	private static long[] _data;

	public static void setup(int maxNumEvents) {
		_eventTypes = DMLScript.OOC_LOG_EVENTS ? new  EventType[maxNumEvents] : null;
		_startTimestamps = DMLScript.OOC_LOG_EVENTS ? new long[maxNumEvents] : null;
		_endTimestamps = DMLScript.OOC_LOG_EVENTS ? new long[maxNumEvents] : null;
		_callerIds = DMLScript.OOC_LOG_EVENTS ? new int[maxNumEvents] : null;
		_threadIds = DMLScript.OOC_LOG_EVENTS ? new long[maxNumEvents] : null;
		_data = DMLScript.OOC_LOG_EVENTS ? new long[maxNumEvents] : null;
	}

	public static int registerCaller(String callerName) {
		int callerId = _callerCtr.incrementAndGet();
		_callerNames.put(callerId, callerName);
		return callerId;
	}

	public static void onComputeEvent(int callerId, long startTimestamp, long endTimestamp) {
		int idx = _logCtr.getAndIncrement();
		_eventTypes[idx] = EventType.COMPUTE;
		_startTimestamps[idx] = startTimestamp;
		_endTimestamps[idx] = endTimestamp;
		_callerIds[idx] = callerId;
		_threadIds[idx] = Thread.currentThread().getId();
	}

	public static void onDiskWriteEvent(int callerId, long startTimestamp, long endTimestamp, long size) {
		int idx = _logCtr.getAndIncrement();
		_eventTypes[idx] = EventType.DISK_WRITE;
		_startTimestamps[idx] = startTimestamp;
		_endTimestamps[idx] = endTimestamp;
		_callerIds[idx] = callerId;
		_threadIds[idx] = Thread.currentThread().getId();
		_data[idx] = size;
	}

	public static void onDiskReadEvent(int callerId, long startTimestamp, long endTimestamp, long size) {
		int idx = _logCtr.getAndIncrement();
		_eventTypes[idx] = EventType.DISK_READ;
		_startTimestamps[idx] = startTimestamp;
		_endTimestamps[idx] = endTimestamp;
		_callerIds[idx] = callerId;
		_threadIds[idx] = Thread.currentThread().getId();
		_data[idx] = size;
	}

	public static void onCacheSizeChangedEvent(int callerId, long timestamp, long cacheSize, long bytesToEvict) {
		int idx = _logCtr.getAndIncrement();
		_eventTypes[idx] = EventType.CACHESIZE_CHANGE;
		_startTimestamps[idx] = timestamp;
		_endTimestamps[idx] = bytesToEvict;
		_callerIds[idx] = callerId;
		_threadIds[idx] = Thread.currentThread().getId();
		_data[idx] = cacheSize;
	}

	public static void putRunSetting(String setting, Object data) {
		_runSettings.put(setting, data);
	}

	public static String getComputeEventsCSV() {
		return getFilteredCSV("ThreadID,CallerID,StartNanos,EndNanos\n", EventType.COMPUTE, false);
	}

	public static String getDiskReadEventsCSV() {
		return getFilteredCSV("ThreadID,CallerID,StartNanos,EndNanos,NumBytes\n", EventType.DISK_READ, true);
	}

	public static String getDiskWriteEventsCSV() {
		return getFilteredCSV("ThreadID,CallerID,StartNanos,EndNanos,NumBytes\n", EventType.DISK_WRITE, true);
	}

	public static String getCacheSizeEventsCSV() {
		return getFilteredCSV("ThreadID,CallerID,Timestamp,ScheduledEvictionSize,CacheSize\n", EventType.CACHESIZE_CHANGE, true);
	}

	private static String getFilteredCSV(String header, EventType filter, boolean data) {
		StringBuilder sb = new StringBuilder();
		sb.append(header);

		int maxIdx = _logCtr.get();
		for (int i = 0; i < maxIdx; i++) {
			if (_eventTypes[i] != filter)
				continue;
			sb.append(_threadIds[i]);
			sb.append(',');
			sb.append(_callerNames.get(_callerIds[i]));
			sb.append(',');
			sb.append(_startTimestamps[i]);
			sb.append(',');
			sb.append(_endTimestamps[i]);
			if (data) {
				sb.append(',');
				sb.append(_data[i]);
			}
			sb.append('\n');
		}

		return sb.toString();
	}

	public static String getRunSettingsCSV() {
		StringBuilder sb = new StringBuilder();
		Set<Map.Entry<String, Object>> entrySet = _runSettings.entrySet();

		int ctr = 0;
		for (Map.Entry<String, Object> entry : entrySet) {
			sb.append(entry.getKey());
			ctr++;
			if (ctr >= entrySet.size())
				sb.append('\n');
			else
				sb.append(',');
		}

		ctr = 0;
		for (Map.Entry<String, Object> entry : _runSettings.entrySet()) {
			sb.append(entry.getValue());
			ctr++;
			if (ctr < entrySet.size())
				sb.append(',');
		}

		return sb.toString();
	}

	public static void clear() {
		_callerCtr.set(0);
		_logCtr.set(0);
		_callerNames.clear();
		_runSettings.clear();
	}

	public enum EventType {
		COMPUTE,
		DISK_WRITE,
		DISK_READ,
		CACHESIZE_CHANGE
	}
}
