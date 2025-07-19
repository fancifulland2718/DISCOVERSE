## 背景信息

Discoverse是一个具身智能机器人仿真平台，基于mujoco的物理引擎，设置了一系列的机器人操作任务。

## 路径信息：
1. 所有路径都是相对于仓库的root路径（在本机中是`/home/tatp/ws/DISCOVERSE`）
2. 机器人/物体资产路径：`models`
3. `models/meshes`保存了机器人和物体的3d模型（obj、stl）
4. `models/mjcf`保存了mjcf格式的机器人和物体的模型
5. `models/mjcf/tasks_airbot_play/`中有多个单臂桌面操作任务
6. `models/mjcf/robot_*.xml`是多种不同的机械臂模型，这些机械臂模型的依赖文件<mujocoinclude>在同级目录下，例如robot_piper.xml的依赖文件在`models/mjcf/agilex_piper`目录中
7. `models`路径在python代码中用`discoverse/__init__.py`中定义的`DISCOVERSE_ASSETS_DIR`来表示

## 问题描述
1. 同一个桌面操作任务要求能够使用的机械臂实现（同一任务，多个不同的机械臂本体），当前的文件组织形式`models/mjcf/tasks_airbot_play`，这些任务只能使用airbot_play这一种机械臂，通用性差，不够灵活

## 任务要求
1. 充分了解mujoco中的[mjcf/xml](https://mujoco.readthedocs.io/en/stable/XMLreference.html)格式的内容和规范
2. 可以参考[robosuite](https://github.com/ARISE-Initiative/robosuite)的相关实现
3. 了解目前的mjcf模型文件（`models/mjcf`下）组织形式，包括物体、机械臂、任务组合的文件组织格式
4. 在`discoverse/envs`中，实现`make_env`的功能，参数包括机械臂模型（机械臂种类包括：`models/mjcf/robot_*.xml`， *所表示的机械臂名称）、任务种类（`models/mjcf/tasks_airbot_play/`中的文件名称），返回一个env对象
5. 这个env对象包含的功能至少包括返回xml对象、xml对象变成string、xml导出为文件。
6. xml文件中的`assets`的路径要求是绝对路径。
7. 测试方式，将导出xml文件，使用`mujoco.MjModel.from_xml_path()`测试能否读取成功。

## 注意
1. 及时删除测试文件
2. 验证之前，制定环境变量`export MUJOCO_GL=egl`
3. **重要**：要求先验证完`mujoco.MjModel.from_xml_path()`，并且没有报错才算任务完成