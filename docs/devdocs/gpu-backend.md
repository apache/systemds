# Initial prototype for GPU backend

A GPU backend implements two important abstract classes:
1. `org.apache.sysml.runtime.controlprogram.context.GPUContext`
2. `org.apache.sysml.runtime.controlprogram.context.GPUObject`

The GPUContext is responsible for GPU memory management and gets call-backs from SystemML's bufferpool on following methods:
1. void acquireRead(MatrixObject mo)
2. void acquireModify(MatrixObject mo)
3. void release(MatrixObject mo, boolean isGPUCopyModified)
4. void exportData(MatrixObject mo)
5. void evict(MatrixObject mo)

A GPUObject (like RDDObject and BroadcastObject) is stored in CacheableData object. It contains following methods that are called back from the corresponding GPUContext:
1. void allocateMemoryOnDevice()
2. void deallocateMemoryOnDevice()
3. long getSizeOnDevice()
4. void copyFromHostToDevice()
5. void copyFromDeviceToHost()

## JCudaContext:
The current prototype supports Nvidia's CUDA libraries using JCuda wrapper. The implementation for the above classes can be found in:
1. `org.apache.sysml.runtime.controlprogram.context.JCudaContext`
2. `org.apache.sysml.runtime.controlprogram.context.JCudaObject`

### Setup instructions for JCudaContext:

1. Install CUDA 7.5
2. Install CuDNN v4 from http://developer.download.nvidia.com/compute/redist/cudnn/v4/cudnn-7.0-win-x64-v4.0-prod.zip
3. Download JCuda binaries version 0.7.5b and JCudnn version 0.7.5. 

* For Windows: Copy the DLLs into C:\lib (or /lib) directory. Link: http://www.jcuda.org/downloads/downloads.html
* For Mac/Linux: TODO !! 