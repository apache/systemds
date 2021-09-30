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

package org.apache.sysds.runtime.transform;

public interface Transformable {


    int getNumRows();
    int getNumColumns();

    /**
     * Returns the double value at the passed row and column.
     * If the value is missing NaN is returned.
     * @param r row of the value
     * @param c column of the value
     */
    double getDoubleValue(int r, int c);

    /**
     * Returns the string of the value at the passed row and column.
     * If the value is missing null is returned.
     * @param r row of the value
     * @param c column of the value
     */
    String getStringValue(int r, int c);
}
