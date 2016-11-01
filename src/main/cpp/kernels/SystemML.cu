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
 
/**********************************
When updating a kernel or adding a new one, 
please compile the ptx file and commit it:
nvcc -ptx SystemML.cu 
***********************************/

// dim => rlen (Assumption: rlen == clen)
// N = length of dense array
extern "C"
__global__ void copyUpperToLowerTriangleDense(double* ret, int dim, int N) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int id_dest = iy * dim + ix;
	if(iy > ix && id_dest < N) {
		// TODO: Potential to reduce the number of threads by half
		int id_src = ix * dim + iy;
		ret[id_dest] = ret[id_src];
	}
}

extern "C"
__device__ double getBoolean(int val) {
	if(val == 0)
		return 0.0;
	else
		return 1.0;
}

// op = {0=plus, 1=minus, 2=multiply, 3=divide, 4=power, 
// 5=less, 6=lessequal, 7=greater, 8=greaterequal, 9=equal, 10=notequal, 
// 11=min, 12=max, 13=and, 14=or, 15=log}
extern "C"
__device__ double binaryOp(double x, double y, int op) {
	// 0=plus, 1=minus, 2=multiply, 3=divide, 4=power
	if(op == 0)
		return x + y;
	else if(op == 1)
		return x - y;
	else if(op == 2)
		return x * y;
	else if(op == 3)
		return x / y;
	else if(op == 4)
		return pow(x, y);
	// 5=less, 6=lessequal, 7=greater, 8=greaterequal, 9=equal, 10=notequal,	
	else if(op == 5) 
		return getBoolean(x < y);
	else if(op == 6)
		return getBoolean(x <= y);
	else if(op == 7)
		return getBoolean(x > y);
	else if(op == 8)
		return getBoolean(x >= y);
	else if(op == 9)
		return getBoolean(x == y);
	else if(op == 10)
		return getBoolean(x != y);
	// 11=min, 12=max, 13=and, 14=or, 15=log
	else if(op == 11) {
		return min(x, y);
	}
	else if(op == 12) {
		return max(x, y);
	}
	return -999;
}

extern "C"
__global__ void dense_matrix_set(double* A,  double scalar, int rlen, int clen) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int index = ix * clen + iy;
	if(index < rlen*clen) {
		A[index] = scalar;
	}	
}

extern "C"
__global__ void dense_matrix_copy(double* A,  double* ret, int rlen, int clen) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int index = ix * clen + iy;
	if(ix < rlen && iy < clen) {
		ret[index] = A[index];
	}
}

// Compares the value and set
extern "C"
__global__ void compareAndSet(double* A,  double* ret, int rlen, int clen, double compareVal, double tol, double ifEqualsVal, double ifLessThanVal, double ifGreaterThanVal) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int index = ix * clen + iy;
	if(ix < rlen && iy < clen) {
		if(abs(A[index]-compareVal) < tol)
			ret[index] = ifEqualsVal;
		else if(A[index] < compareVal)
			ret[index] = ifLessThanVal;
		else		
			ret[index] = ifGreaterThanVal;
	}
}

extern "C"
__global__ void binCellOp(double* A, double* B, double* C, 
	int maxRlen, int maxClen, int vectorAStatus, int vectorBStatus, int op) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(ix < maxRlen && iy < maxClen) {
		int outIndex = ix * maxClen + iy;
		int aIndex = outIndex;
		int bIndex = outIndex;
		if(vectorAStatus == 1)
			aIndex = ix; // clen == 1
		else if(vectorAStatus == 2)
			aIndex = iy; // rlen == 1
		if(vectorBStatus == 1)
			bIndex = ix; // clen == 1
		else if(vectorBStatus == 2)
			bIndex = iy; // rlen == 1
		C[outIndex] = binaryOp(A[aIndex], B[bIndex], op);
		// printf("C[%d] = A[%d](%f) B[%d](%f) (%d %d)\n", outIndex, aIndex, A[aIndex], bIndex,  B[bIndex], (ix+1), (iy+1));
	}
}

extern "C"
__global__ void binCellScalarOp(double* A, double scalar, double* C, int rlenA, int clenA, int op, int isLeftScalar) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int index = ix * clenA + iy;
	if(index < rlenA*clenA) {
		if(isLeftScalar)
			C[index] = binaryOp(scalar, A[index], op);
		else
			C[index] = binaryOp(A[index], scalar, op);
	}
}