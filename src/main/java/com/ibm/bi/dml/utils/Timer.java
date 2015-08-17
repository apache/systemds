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

package com.ibm.bi.dml.utils;

public class Timer 
{

	
	long start;
	double sofar;
	
	public Timer() {
		start = 0 ;
		sofar = 0.0;
	}
	
	public void start() {
		start = System.nanoTime();
		sofar = 0.0;
	}
	
	public double stop() {
		double duration = sofar + (System.nanoTime()-start)*1e-6;
		sofar = 0.0;
		start = 0;
		return duration;
	}
	
	public double nanostop() {
		double duration = sofar + (System.nanoTime()-start);
		sofar = 0.0;
		start = 0;
		return duration;
	}
	
	public double pause() {
		sofar += (System.nanoTime()-start)*1e-6;
		return sofar;
	}
	
	public void resume() {
		start = System.nanoTime();
	}
	
}
