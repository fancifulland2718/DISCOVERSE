"""
Mink逆运动学求解器

基于mink库的通用逆运动学求解器，支持多种机械臂。
"""

import numpy as np
import mujoco
from typing import Optional, Tuple, List, Dict, Any
from scipy.spatial.transform import Rotation

try:
    import mink
except ImportError:
    print("Warning: mink library not found. Please install mink for IK solving.")
    mink = None

from .robot_config import RobotConfigLoader

from .robot_config import RobotConfigLoader

class MinkIKSolver:
    """基于Mink的通用逆运动学求解器"""
    
    def __init__(self, robot_config: RobotConfigLoader, mj_model: mujoco.MjModel, mj_data: mujoco.MjData = None):
        """
        初始化Mink IK求解器
        
        Args:
            robot_config: 机械臂配置加载器
            mj_model: MuJoCo模型
            mj_data: MuJoCo数据（用于调试）
            
        Raises:
            ImportError: mink库未安装
            ValueError: 配置无效
        """
        if mink is None:
            raise ImportError("mink library is required for IK solving. Please install mink.")
        
        self.robot_config = robot_config
        self.mj_model = mj_model
        self.mj_data = mj_data  # 添加这个用于调试
        
        # 初始化mink配置
        self.configuration = mink.Configuration(mj_model)
        
        # 获取求解器配置 - 支持多种配置键名
        ik_solver_config = robot_config.config.get('ik_solver', {})
        if not ik_solver_config:
            ik_solver_config = robot_config.config.get('mink_ik', {})
        self.solver_config = ik_solver_config
        self.solver_type = self.solver_config.get('solver_type', 'quadprog')
        self.position_tolerance = self.solver_config.get('position_tolerance', 1e-4)
        self.orientation_tolerance = self.solver_config.get('orientation_tolerance', 1e-4)
        self.max_iterations = self.solver_config.get('max_iterations', 50)
        self.damping = float(self.solver_config.get('damping', 1e-3))  # 确保是数字类型
        self.dt = self.solver_config.get('dt', 2e-3)  # 积分时间步长
        
        # 调试信息：打印当前使用的容差
        print(f"🔧 Mink IK Solver 初始化:")
        print(f"   位置容差: {self.position_tolerance}")
        print(f"   姿态容差: {self.orientation_tolerance}")
        print(f"   最大迭代: {self.max_iterations}")
        
        # 设置IK任务
        self._setup_ik_tasks()
        
        # 性能统计
        self.solve_count = 0
        self.solve_times = []
        self.convergence_count = 0
        
    def _setup_ik_tasks(self):
        """设置IK任务"""
        # 末端执行器位置和姿态任务
        self.end_effector_task = mink.FrameTask(
            frame_name=self.robot_config.end_effector_site,
            frame_type="site",
            position_cost=self.solver_config.get('position_cost', 1.0),
            orientation_cost=self.solver_config.get('orientation_cost', 1.0),
            lm_damping=self.damping,
        )
        
        # 姿态任务（保持关节在舒适位置）
        self.posture_task = mink.PostureTask(
            model=self.mj_model,
            cost=self.solver_config.get('posture_cost', 1e-2)
        )
        
        # 任务列表
        self.tasks = [self.end_effector_task, self.posture_task]
        
        # 设置姿态任务的目标为home位姿
        # 使用MuJoCo模型的完整qpos，而不只是机械臂部分
        if self.mj_model.nkey > 0:
            # 使用keyframe中的home位置（这包含完整的模型状态）
            home_qpos = self.mj_model.key(0).qpos.copy()
            self.configuration.update(home_qpos)
            self.posture_task.set_target_from_configuration(self.configuration)
        else:
            # 备选方案：创建完整的qpos
            home_pose = self.robot_config.get_home_pose()
            if home_pose is not None:
                # 创建完整模型的qpos
                full_model_qpos = np.zeros(self.mj_model.nq)
                
                # 设置机械臂部分（假设前self.robot_config.dof个是机械臂关节）
                robot_dof = min(len(home_pose), self.robot_config.dof)
                full_model_qpos[:robot_dof] = home_pose[:robot_dof]
                
                # 设置夹爪为打开状态
                gripper_indices = self.robot_config.gripper_joint_indices
                for idx in gripper_indices:
                    if idx < len(full_model_qpos):
                        full_model_qpos[idx] = self.robot_config.gripper_range[1]
                
                self.configuration.update(full_model_qpos)
                self.posture_task.set_target_from_configuration(self.configuration)
    
    def solve_ik(self, 
                 target_pos: np.ndarray, 
                 target_ori: np.ndarray, 
                 current_qpos: np.ndarray,
                 reference_qpos: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        求解逆运动学
        
        Args:
            target_pos: 目标位置 [x, y, z]
            target_ori: 目标姿态矩阵 (3x3) 或四元数 [qw, qx, qy, qz]
            current_qpos: 当前关节位置
            reference_qpos: 参考关节位置（用于选择最优解）
            
        Returns:
            Tuple[关节位置, 是否收敛, 求解信息]
        """
        import time
        start_time = time.time()
        
        try:
            # 更新当前配置
            self.configuration.update(current_qpos)
            
            # 处理目标姿态
            if target_ori.shape == (4,):
                # 四元数转旋转矩阵
                target_rot_matrix = Rotation.from_quat(target_ori[[1, 2, 3, 0]]).as_matrix()
            elif target_ori.shape == (3, 3):
                target_rot_matrix = target_ori
            else:
                raise ValueError(f"Invalid target orientation shape: {target_ori.shape}")
            
            # 构建目标变换矩阵
            T_target = np.eye(4)
            T_target[:3, :3] = target_rot_matrix
            T_target[:3, 3] = target_pos
            
            # 调试信息：打印目标姿态（只在有mj_data时）
            if self.mj_data is not None:
                try:
                    current_site_xmat = self.mj_data.site_xmat[self.mj_model.site(self.robot_config.end_effector_site).id].reshape(3, 3)
                    print(f"   🤖 当前末端姿态:")
                    print(f"      X轴: {current_site_xmat[:, 0]}")
                    print(f"      Y轴: {current_site_xmat[:, 1]}")
                    print(f"      Z轴: {current_site_xmat[:, 2]}")
                    print(f"   🎯 目标末端姿态:")
                    print(f"      X轴: {target_rot_matrix[:, 0]}")
                    print(f"      Y轴: {target_rot_matrix[:, 1]}")
                    print(f"      Z轴: {target_rot_matrix[:, 2]}")
                except:
                    pass
            
            # 设置目标
            target_SE3 = mink.SE3.from_matrix(T_target)
            self.end_effector_task.set_target(target_SE3)
            
            # 如果提供了参考位置，更新姿态任务
            if reference_qpos is not None:
                temp_config = mink.Configuration(self.mj_model)
                temp_config.update(reference_qpos)
                self.posture_task.set_target_from_configuration(temp_config)
            
            # 迭代求解
            dt = 1e-3
            converged = False
            iteration = 0
            errors = []
            
            for iteration in range(self.max_iterations):
                # 计算速度
                velocity = mink.solve_ik(
                    self.configuration, 
                    self.tasks, 
                    dt, 
                    self.solver_type, 
                    self.damping
                )
                
                # 积分更新配置
                self.configuration.integrate_inplace(velocity, dt)
                
                # 检查收敛
                error = self.end_effector_task.compute_error(self.configuration)
                position_error = np.linalg.norm(error[:3])
                orientation_error = np.linalg.norm(error[3:])
                
                errors.append({
                    'position_error': position_error,
                    'orientation_error': orientation_error,
                    'total_error': position_error + orientation_error
                })
                
                # 检查收敛条件
                if (position_error < self.position_tolerance and 
                    orientation_error < self.orientation_tolerance):
                    converged = True
                    break
                    
                # 调试信息：打印最后几次迭代的错误
                if iteration >= self.max_iterations - 5:
                    print(f"   迭代 {iteration}: pos_err={position_error:.6f} (tol={self.position_tolerance}), ori_err={orientation_error:.6f} (tol={self.orientation_tolerance})")
            
            # 获取解
            solution = self.configuration.q[:self.robot_config.arm_joints_count].copy()
            
            # 验证解的有效性
            is_valid = self._validate_solution(solution)
            
            # 记录统计信息
            solve_time = time.time() - start_time
            self.solve_count += 1
            self.solve_times.append(solve_time)
            if converged:
                self.convergence_count += 1
            
            # 构建求解信息
            solve_info = {
                'converged': converged and is_valid,
                'iterations': iteration + 1,
                'solve_time': solve_time,
                'final_position_error': errors[-1]['position_error'] if errors else float('inf'),
                'final_orientation_error': errors[-1]['orientation_error'] if errors else float('inf'),
                'is_valid_solution': is_valid,
                'error_history': errors
            }
            
            return solution, converged and is_valid, solve_info
            
        except Exception as e:
            solve_time = time.time() - start_time
            error_info = {
                'converged': False,
                'iterations': 0,
                'solve_time': solve_time,
                'error': str(e),
                'final_position_error': float('inf'),
                'final_orientation_error': float('inf'),
                'is_valid_solution': False
            }
            
            # 返回当前关节位置作为fallback
            fallback_solution = current_qpos[:self.robot_config.arm_joints_count].copy()
            return fallback_solution, False, error_info
    
    def solve_ik_position_only(self, 
                              target_pos: np.ndarray,
                              current_qpos: np.ndarray,
                              keep_orientation: bool = True) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        仅求解位置的逆运动学（保持当前姿态）
        
        Args:
            target_pos: 目标位置
            current_qpos: 当前关节位置
            keep_orientation: 是否保持当前姿态
            
        Returns:
            Tuple[关节位置, 是否收敛, 求解信息]
        """
        if keep_orientation:
            # 获取当前末端执行器姿态
            current_T = self._forward_kinematics(current_qpos)
            current_ori = current_T[:3, :3]
        else:
            # 使用默认姿态
            current_ori = np.eye(3)
        
        return self.solve_ik(target_pos, current_ori, current_qpos)
    
    def _forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        计算正运动学
        
        Args:
            joint_positions: 关节位置
            
        Returns:
            末端执行器变换矩阵 (4x4)
        """
        # 创建临时数据结构
        temp_data = mujoco.MjData(self.mj_model)
        temp_data.qpos[:len(joint_positions)] = joint_positions
        mujoco.mj_forward(self.mj_model, temp_data)
        
        # 获取末端执行器位置和姿态
        site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.robot_config.end_effector_site)
        pos = temp_data.site_xpos[site_id]
        mat = temp_data.site_xmat[site_id].reshape(3, 3)
        
        # 构建变换矩阵
        T = np.eye(4)
        T[:3, :3] = mat
        T[:3, 3] = pos
        
        return T
    
    def _validate_solution(self, joint_positions: np.ndarray) -> bool:
        """
        验证IK解的有效性
        
        Args:
            joint_positions: 关节位置
            
        Returns:
            是否有效
        """
        # 检查关节限制
        if not self.robot_config.validate_joint_position(joint_positions):
            return False
        
        # 检查是否存在NaN或无穷大
        if np.any(np.isnan(joint_positions)) or np.any(np.isinf(joint_positions)):
            return False
        
        # 可以添加更多验证规则（如奇异性检查、碰撞检查等）
        
        return True
    
    def get_jacobian(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        计算雅可比矩阵
        
        Args:
            joint_positions: 关节位置
            
        Returns:
            雅可比矩阵 (6 x n_joints)
        """
        # 更新配置
        self.configuration.update(joint_positions)
        
        # 计算雅可比
        jacobian = np.zeros((6, self.robot_config.arm_joints_count))
        
        # 获取站点ID
        site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.robot_config.end_effector_site)
        
        # 创建临时数据
        temp_data = mujoco.MjData(self.mj_model)
        temp_data.qpos[:len(joint_positions)] = joint_positions
        mujoco.mj_forward(self.mj_model, temp_data)
        
        # 计算雅可比矩阵
        jacp = np.zeros(3 * self.mj_model.nv)  # 位置雅可比
        jacr = np.zeros(3 * self.mj_model.nv)  # 旋转雅可比
        
        mujoco.mj_jacSite(self.mj_model, temp_data, jacp, jacr, site_id)
        
        # 重塑并截取相关部分
        jacp = jacp.reshape(3, -1)[:, :self.robot_config.arm_joints_count]
        jacr = jacr.reshape(3, -1)[:, :self.robot_config.arm_joints_count]
        
        jacobian[:3, :] = jacp
        jacobian[3:, :] = jacr
        
        return jacobian
    
    def check_reachability(self, target_pos: np.ndarray, workspace_type: str = 'reachable') -> bool:
        """
        检查目标位置是否可达
        
        Args:
            target_pos: 目标位置
            workspace_type: 工作空间类型
            
        Returns:
            是否可达
        """
        return self.robot_config.is_position_in_workspace(target_pos, workspace_type)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取求解器统计信息
        
        Returns:
            统计信息字典
        """
        if self.solve_count == 0:
            return {
                'total_solves': 0,
                'convergence_rate': 0.0,
                'average_solve_time': 0.0,
                'average_iterations': 0.0
            }
        
        return {
            'total_solves': self.solve_count,
            'convergence_rate': self.convergence_count / self.solve_count,
            'average_solve_time': np.mean(self.solve_times),
            'min_solve_time': np.min(self.solve_times),
            'max_solve_time': np.max(self.solve_times),
            'solver_type': self.solver_type,
            'position_tolerance': self.position_tolerance,
            'orientation_tolerance': self.orientation_tolerance
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.solve_count = 0
        self.solve_times = []
        self.convergence_count = 0
    
    def set_posture_target(self, target_qpos: np.ndarray):
        """
        设置姿态任务的目标
        
        Args:
            target_qpos: 目标关节位置
        """
        temp_config = mink.Configuration(self.mj_model)
        temp_config.update(target_qpos)
        self.posture_task.set_target_from_configuration(temp_config)
    
    def __str__(self) -> str:
        """字符串表示"""
        stats = self.get_statistics()
        return (f"MinkIKSolver({self.robot_config.robot_name}, "
                f"convergence_rate={stats['convergence_rate']:.2f}, "
                f"avg_time={stats['average_solve_time']:.4f}s)")
    
    def __repr__(self) -> str:
        """对象表示"""
        return self.__str__()


def create_mink_solver(robot_config: RobotConfigLoader, mj_model: mujoco.MjModel, mj_data: mujoco.MjData = None) -> MinkIKSolver:
    """
    便利函数：创建Mink IK求解器
    
    Args:
        robot_config: 机械臂配置
        mj_model: MuJoCo模型
        mj_data: MuJoCo数据（可选，用于调试）
        
    Returns:
        Mink IK求解器实例
    """
    return MinkIKSolver(robot_config, mj_model, mj_data) 