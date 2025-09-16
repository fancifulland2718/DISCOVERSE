import numpy as np

class AirbotPlayIK:
    """
    AirbotPlayIK 类提供逆向运动学（IK）的求解函数，
    用于根据目标位置和姿态计算机械臂各关节的角度。
    
    类中定义了一些机械臂的几何参数（连杆长度）和关节运动范围，
    以及一个偏置数组用于从计算出的角度中进行调整，使得求解结果更符合实际机械臂的零位配置。
    """

    # 预设关节角度偏置，用于对计算出的逆运动学解进行调整
    bias = np.array([0.0, -2.7549, 2.7549, 1.5708, 0.0, 0.0])
    # 连杆 a1：机器人基座与第一个关节之间的长度
    a1 = 0.1172
    # 连杆 a3：机器人第二段连杆长度
    a3 = 0.27009
    # 连杆 a4：机器人第三段连杆长度
    a4 = 0.29015
    # 连杆 a6：用于关节6与工具末端的偏移
    a6 = 0.23645
    # 机械臂各个关节可运动范围，二维数组第一行为下边界，第二行为上边界
    arm_joint_range = np.array([
        [-3.151 , -2.963 , -0.094, -3.012 , -1.859 , -3.017 ],
        [ 2.089 ,  0.181 ,  3.161,  3.012 ,  1.859 ,  3.017 ]
    ])
    # 各个关节运动范围的跨度
    joint_range_scale = arm_joint_range[1] - arm_joint_range[0]

    # 定义旋转矩阵，用于将姿态转换至机械臂基坐标系下的统一表示，
    # 此矩阵常用于 properIK 函数中调整目标姿态
    arm_rot_mat = np.array([
        [ 0., -0.,  1.],
        [ 0.,  1.,  0.],
        [-1.,  0.,  0.]
    ])

    def __init__(self) -> None:
        # 构造函数，无需额外初始化操作
        pass

    def properIK(self, pos, ori, ref_q=None):
        """
        properIK 为逆运动学求解入口函数。
        提供目标末端执行器在世界坐标中的位置 pos 及姿态 ori，
        内部会根据 arm_rot_mat 对姿态进行转换，再调用 inverseKin 求解 IK 问题。
        
        参数:
            pos (array): 目标位置，3 维向量
            ori (ndarray): 目标姿态，3x3 旋转矩阵
            ref_q (array, optional): 参考关节值，用于选择最优 IK 解
            
        返回:
            如果提供 ref_q，则返回与之最接近的单一 IK 解；
            否则，返回所有满足关节范围要求的逆运动学解列表。
        """
        return self.inverseKin(pos, ori @ self.arm_rot_mat, ref_q)

    def inverseKin(self, pos, ori, ref_q=None):
        """
        根据给定的末端执行器位置 (pos) 和姿态 (ori) 求解机械臂的各关节角度，
        同时考虑关节运动范围和预设偏置。
        
        参数:
            pos (array): 目标位置，3 维向量
            ori (ndarray): 目标姿态，3x3 旋转矩阵
            ref_q (array, optional): 参考关节角度，用于选择最优解
            
        返回:
            满足关节运动范围的 IK 解（或多个解）。
            
        抛出:
            如果没有任何 IK 解满足条件，则抛出 ValueError。
        """
        assert len(pos) == 3 and ori.shape == (3,3)
        # 调整位置，将第6关节的影响转移到第5关节（考虑工具末端偏移）
        pos = self.move_joint6_2_joint5(pos, ori)
        # 存储关节角度的列表，初始预留6个关节角
        angle = [0.0] * 6
        ret = []

        # 第一个关节有两个可能的解（正解或对称解）
        for i1 in [1, -1]:
            angle[0] = np.arctan2(i1 * pos[1], i1 * pos[0])
            # 计算第三个关节角度的余弦值，利用余弦定理求解
            c3 = (pos[0] ** 2 + pos[1] ** 2 + (pos[2] - self.a1) ** 2 - self.a3 ** 2 - self.a4 ** 2) / (2 * self.a3 * self.a4)
            if c3 > 1 or c3 < -1:
                raise ValueError("Fail to solve inverse kinematics: pos={}, ori={}".format(pos, ori))

            # 第三关节的正负两个可能的解
            for i2 in [1, -1]:
                s3 = i2 * np.sqrt(1 - c3 ** 2)
                angle[2] = np.arctan2(s3, c3)
                # 计算辅助变量 k1 和 k2，用于求解第二关节角度
                k1 = self.a3 + self.a4 * c3
                k2 = self.a4 * s3
                # 通过三角函数计算第二关节角度
                angle[1] = np.arctan2(k1 * (pos[2] - self.a1) - i1 * k2 * np.sqrt(pos[0] ** 2 + pos[1] ** 2),
                                   i1 * k1 * np.sqrt(pos[0] ** 2 + pos[1] ** 2) + k2 * (pos[2] - self.a1))
                # 构造一个 3x3 旋转矩阵 R，其代表前3个关节构型对末端方向的影响
                R = np.array([
                    [np.cos(angle[0]) * np.cos(angle[1] + angle[2]),
                     -np.cos(angle[0]) * np.sin(angle[1] + angle[2]),
                     np.sin(angle[0])],
                    [np.sin(angle[0]) * np.cos(angle[1] + angle[2]),
                     -np.sin(angle[0]) * np.sin(angle[1] + angle[2]),
                     -np.cos(angle[0])],
                    [np.sin(angle[1] + angle[2]), np.cos(angle[1] + angle[2]), 0]
                ])
                # 计算旋转差 ori1，该矩阵用于后续关节角度（关节 4、5、6）的求解
                ori1 = R.T @ ori
                # 关节 4、5、6 的求解，i5 控制第二组正负可能性
                for i5 in [1, -1]:
                    angle[3] = np.arctan2(i5 * ori1[2, 2], i5 * ori1[1, 2])
                    angle[4] = np.arctan2(i5 * np.sqrt(ori1[2, 2] ** 2 + ori1[1, 2] ** 2), ori1[0, 2])
                    angle[5] = np.arctan2(-i5 * ori1[0, 0], -i5 * ori1[0, 1])
                    # 对求解出的角度加上偏置，并执行角度归一化
                    js = self.add_bias(angle)
                    # 判断求得的IK解是否在各关节允许的运动范围之内
                    if np.all((js > self.arm_joint_range[0]) * (js < self.arm_joint_range[1])):
                        ret.append(js)
        if len(ret) == 0:
            raise ValueError("Fail to solve inverse kinematics: pos={}, ori={}".format(pos, ori))

        # 如果给定了参考解，则选择距离参考解最近的那组关节角
        if ref_q is not None:
            joint_dist_lst = []
            for js in ret:
                joint_dist_lst.append(np.sum(np.abs(ref_q - js) / self.joint_range_scale))
            q = ret[np.argmin(joint_dist_lst)]
            return q
        else:
            return ret

    def add_bias(self, angle):
        """
        对输入的关节角度列表加上偏置，并将角度归一化到 [-π, π] 之间。
        
        参数:
            angle (list): 求解出的原始关节角度列表
        
        返回:
            带偏置调整后的角度列表
        """
        ret = []
        for i in range(len(angle)):
            a = angle[i] + self.bias[i]
            while a > np.pi:
                a -= 2 * np.pi
            while a < -np.pi:
                a += 2 * np.pi
            ret.append(a)
        return ret

    def move_joint6_2_joint5(self, pos, ori):
        """
        将末端执行器位置从关节6的坐标系调整到关节5的坐标系，
        考虑工具末端（连杆 a6）的偏移作用。
        
        参数:
            pos (array): 初始末端位置
            ori (ndarray): 末端姿态的旋转矩阵
        
        返回:
            调整后的末端位置
        """
        ret = np.array([
            -ori[0, 2] * self.a6 + pos[0],
            -ori[1, 2] * self.a6 + pos[1],
            -ori[2, 2] * self.a6 + pos[2]
        ])
        return ret

    def j3_ik(self, pos):
        """
        提供3关节逆向运动学求解，仅对机械臂前三个关节进行求解，
        可用于某些简化场景下的初步运动规划。
        
        参数:
            pos (array): 目标位置
        
        返回:
            满足条件的 IK 解列表（仅3个关节）
        """
        angle = [0.0] * 3
        ret = []

        for i1 in [1, -1]:
            angle[0] = np.arctan2(i1 * pos[1], i1 * pos[0])
            c3 = (pos[0] ** 2 + pos[1] ** 2 + (pos[2] - self.a1) ** 2 - self.a3 ** 2 - self.a4 ** 2) / (2 * self.a3 * self.a4)
            if c3 > 1 or c3 < -1:
                raise ValueError("Fail to solve inverse kinematics")

            for i2 in [1, -1]:
                s3 = i2 * np.sqrt(1 - c3 ** 2)
                angle[2] = np.arctan2(s3, c3)
                k1 = self.a3 + self.a4 * c3
                k2 = self.a4 * s3
                angle[1] = np.arctan2(k1 * (pos[2] - self.a1) - i1 * k2 * np.sqrt(pos[0] ** 2 + pos[1] ** 2),
                                   i1 * k1 * np.sqrt(pos[0] ** 2 + pos[1] ** 2) + k2 * (pos[2] - self.a1))
                js = self.add_bias(angle)
                # 检查前三个关节的解是否在指定范围内
                if np.all((js > self.arm_joint_range[0,:3]) * (js < self.arm_joint_range[1,:3])):
                    ret.append(js)
        return ret

if __name__ == "__main__":
    # 测试代码：通过给定位姿求解机械臂关节角度，并打印所有求解结果
    arm_ik = AirbotPlayIK()

    # 目标末端位置
    trans = np.array([ 0.276, -0., 0.219])
    # 目标末端姿态，单位矩阵表示无旋转
    rot   = np.array([
        [1., -0., -0.],
        [0.,  1., -0.],
        [0.,  0.,  1.]
    ])

    qs = arm_ik.properIK(trans, rot)

    for q in qs:
        print(q)