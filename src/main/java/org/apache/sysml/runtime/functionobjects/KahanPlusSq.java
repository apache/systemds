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

package org.apache.sysml.runtime.functionobjects;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.KahanObject;

import java.io.Serializable;


/**
 * Runtime function to perform the summation of squared values using
 * the Kahan summation algorithm.
 */
public class KahanPlusSq extends KahanFunction implements Serializable {

    private static final long serialVersionUID = 5774388904472231717L;

    private static KahanPlusSq singleObj = null;
    private KahanPlus kplus = null;

    private KahanPlusSq() {
        kplus = KahanPlus.getKahanPlusFnObject();
    }

    /**
     * Get the KahanPlusSq singleton object.
     */
    public static KahanPlusSq getKahanPlusSqFnObject() {
        if (singleObj == null)
            singleObj = new KahanPlusSq();
        return singleObj;
    }

    public Object clone() throws CloneNotSupportedException {
        // cloning is not supported for singleton classes
        throw new CloneNotSupportedException();
    }

    /**
     * Square the given term, then add to the existing sum using
     * the Kahan summation algorithm.
     *
     * @param kObj A KahanObject supplied as a Data type containing
     *             the current sum and correction factor for the
     *             Kahan summation algorithm.
     * @param in The current term to be squared and added.
     */
    @Override
    public Data execute(Data kObj, double in)
            throws DMLRuntimeException {
        kplus.execute(kObj, in * in);
        return kObj;
    }

    /**
     * Add the given sum and correction factor to the existing
     * sum in the KahanObject using the Kahan summation algorithm.
     *
     * @param kObj A KahanObject supplied as a Data type containing
     *             the current sum and correction factor for the
     *             Kahan summation algorithm.
     * @param sum The current sum.
     * @param corr The current correction factor.
     */
    @Override
    public Data execute(Data kObj, double sum, double corr)
            throws DMLRuntimeException {
        kplus.execute(kObj, sum, corr);
        return kObj;
    }

    @Override
    public void execute2(KahanObject kObj, double in) {
        kplus.execute2(kObj, in * in);
    }
    
    @Override
    public void execute3(KahanObject kObj, double in, int count) {
        kplus.execute3(kObj, in * in, count);
    }
}
