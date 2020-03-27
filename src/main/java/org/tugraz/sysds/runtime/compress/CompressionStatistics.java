/*
 * Modifications Copyright 2020 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.runtime.compress;

public class CompressionStatistics {
	public double timePhase1 = -1;
	public double timePhase2 = -1;
	public double timePhase3 = -1;
	public double timePhase4 = -1;
	public double timePhase5 = -1;
	public double estSize = -1;
	public double size = -1;
	public double ratio = -1;

	public CompressionStatistics() {
		// do nothing
	}

	public CompressionStatistics(double t1, double t2, double t3, double t4, double t5) {
		timePhase1 = t1;
		timePhase2 = t2;
		timePhase3 = t3;
		timePhase4 = t4;
		timePhase5 = t5;
	}
}
