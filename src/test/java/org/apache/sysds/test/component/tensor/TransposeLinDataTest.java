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

package org.apache.sysds.test.component.tensor;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.data.TensorBlock;
 import java.util.Arrays;

public class TransposeLinDataTest {

    @Test
    public void Testrightelem(){
        int[] shape = {2, 3, 4};
        TensorBlock tensor = TensorUtils.createArangeTensor(shape);

        Assert.assertArrayEquals(new int[]{2, 3, 4}, tensor.getDims()); 
        Assert.assertEquals(0.0, tensor.get(new int[]{0, 0, 0}));
        Assert.assertEquals(23.0, tensor.get(new int[]{1, 2, 3}));
        Assert.assertEquals(6.0, tensor.get(new int[]{0, 1, 2}));
        Assert.assertEquals(12.0, tensor.get(new int[]{1, 0, 0}));
        printTensor(tensor);


        int[] permutation = {1, 0, 2};
        TensorBlock outTensor = PermuteIt.permute(tensor, permutation); 
        printTensor(outTensor); 

        Assert.assertArrayEquals(new int[]{3, 2, 4}, outTensor.getDims()); 
        Assert.assertEquals(0.0, outTensor.get(new int[]{0,0,0})); 
        Assert.assertEquals(23.0, outTensor.get(new int[]{2, 1, 3})); 
        Assert.assertEquals(12.0, outTensor.get(new int[]{0, 1, 0})); 
        Assert.assertEquals(17.0, outTensor.get(new int[]{1, 1, 1})); 
        

        int[] second_permutation = {2, 1, 0}; 
        TensorBlock perm2Block = PermuteIt.permute(tensor, second_permutation); 
        printTensor(perm2Block); 

        Assert.assertArrayEquals(new int[]{4, 3, 2}, perm2Block.getDims()); 
        Assert.assertEquals(0.0, perm2Block.get(new int[]{0, 0, 0}));
        Assert.assertEquals(12.0, perm2Block.get(new int[]{0, 0, 1})); 
        Assert.assertEquals(11.0, perm2Block.get(new int[]{3, 2, 0})); 
        Assert.assertEquals(23.0, perm2Block.get(new int[]{3, 2, 1})); 
        
    }

    


    public class TensorUtils {

        public static TensorBlock createArangeTensor(int[] shape) {
            TensorBlock tb = new TensorBlock(ValueType.FP64, shape);
            tb.allocateBlock();
            double[] counter = { 0.0 };
            int[] currentIndices = new int[shape.length];
            
            fillRecursively(tb, shape, 0, currentIndices, counter);
            
            return tb;
        }

        private static void fillRecursively(TensorBlock tb, int[] shape, int dim, int[] currentIndices, double[] counter) {
            if (dim == shape.length) {
                tb.set(currentIndices, counter[0]);
                counter[0]++; 
                return;
            }

            for (int i = 0; i < shape[dim]; i++) {
                currentIndices[dim] = i;

                fillRecursively(tb, shape, dim + 1, currentIndices, counter);
            }
        }
    }



    public class PermuteIt {


        public static TensorBlock permute(TensorBlock tensor, int[] permute_dims) { 

            int anz_dims = tensor.getNumDims(); 
            int[] dims = tensor.getDims();
            ValueType tensorType = tensor.getValueType();

            int[] out_shape = new int[anz_dims]; 

            for (int idx = 0; idx < anz_dims; idx++){
                out_shape[idx] = dims[permute_dims[idx]];
            }

            TensorBlock outTensor = new TensorBlock(tensorType, out_shape); 
            outTensor.allocateBlock();

            int[] inIndex = new int[anz_dims]; 
            int[] outIndex = new int[anz_dims]; 

            rekursion(tensor, outTensor, permute_dims, dims, 0, inIndex, outIndex); 
            return outTensor; 
        }   

        public static void rekursion(TensorBlock inTensor, 
                                     TensorBlock outTensor, 
                                     int[] permutation, 
                                     int[] inShape, 
                                     int dim, 
                                     int[] inIndex, 
                                     int[]outIndex
                                     ){

            if (dim == inShape.length) {
                for(int idx = 0; idx < permutation.length; idx++){
                    outIndex[idx] = inIndex[permutation[idx]]; 
                }
                double val = (double) inTensor.get(inIndex); 
                outTensor.set(outIndex, val); 
                return; 
            }

            for(int idx = 0; idx < inShape[dim]; idx++){
                inIndex[dim] = idx; 
                rekursion(inTensor, outTensor, permutation, inShape, dim+1, inIndex, outIndex);
            }
            
        }

    }
   

    public static void printTensor(TensorBlock tb) {
        StringBuilder sb = new StringBuilder();
        int[] shape = tb.getDims();
        int[] currentIndices = new int[shape.length];
        
        sb.append("Tensor(").append(Arrays.toString(shape)).append("):\n");
        printRecursive(tb, shape, 0, currentIndices, sb, 0);
        
        System.out.println(sb.toString());
    }

    private static void printRecursive(TensorBlock tb, int[] shape, int dim, int[] indices, StringBuilder sb, int indent) {
        for (int k = 0; k < indent; k++) sb.append(" ");

        sb.append("[");

        if (dim == shape.length - 1) {
            for (int i = 0; i < shape[dim]; i++) {
                indices[dim] = i;
                double val = (double) tb.get(indices); 
                sb.append(String.format("%.1f", val)); 
                if (i < shape[dim] - 1) sb.append(", ");
            }
            sb.append("]");
        } 

        else {
            sb.append("\n");
            for (int i = 0; i < shape[dim]; i++) {
                indices[dim] = i;
                printRecursive(tb, shape, dim + 1, indices, sb, indent + 2);
                
                if (i < shape[dim] - 1) {
                    sb.append(",");
                    sb.append("\n"); 
                    if (shape.length - dim > 2) sb.append("\n"); 
                }
            }
            sb.append("\n"); 
            for (int k = 0; k < indent; k++) sb.append(" ");
            sb.append("]");
        }
    }

}