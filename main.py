import numpy as np
import os
import time
import subprocess
from systemds.context import SystemDSContext

# 确保路径存在
project_path = "D:\\PythonProject1\\pythonAPI\\pythonAPI"
os.makedirs(project_path, exist_ok=True)

# 创建一个随机的 Numpy 数据集
data = np.random.rand(1000, 1000)

# 使用 Python API 传输数据
with SystemDSContext() as sds:
    start_time = time.time()
    sds_data = sds.from_numpy(data)
    end_time = time.time()
    print(f"数据传输时间 (Python API): {end_time - start_time} 秒")

# 创建一个 DML 脚本来处理相同的数据集
dml_script = """
D = read($data, data_type="matrix");
write(D, $out_file);
"""

# 保存 DML 脚本到文件
dml_script_path = os.path.join(project_path, "code.dml")
with open(dml_script_path, "w") as file:
    file.write(dml_script)

# 保存数据到磁盘
data_csv_path = os.path.join(project_path, "data.csv")
output_csv_path = os.path.join(project_path, "output.csv")
np.savetxt(data_csv_path, data, delimiter=",")

# 通过 SystemDS 直接执行 DML 脚本
start_time = time.time()
subprocess.run(["D:/systemds-3.2.0-bin/bin/systemds", dml_script_path, "-args", f"data={data_csv_path}", f"out_file={output_csv_path}"])
end_time = time.time()
print(f"数据传输时间 (DML 脚本): {end_time - start_time} 秒")
