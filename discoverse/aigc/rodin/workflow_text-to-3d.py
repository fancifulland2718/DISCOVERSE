import os
import time
import json
import requests
import argparse

RODIN_API_KEY = os.getenv("RODIN_API_KEY")  # 替换为你的实际API密钥

API_ENDPOINT = "https://api.hyper3d.com/api/v2/rodin"

def create_session():
    """创建HTTP session"""
    session = requests.Session()
    
    # 设置User-Agent
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    })
    
    return session

def load_task_info(task_file):
    """读取任务信息文件"""
    if os.path.exists(task_file):
        try:
            with open(task_file, "r", encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"读取任务文件失败: {e}")
            return {}
    return {}

def save_task_info(task_file, task_info):
    """增量保存任务信息"""
    existing_tasks = load_task_info(task_file)
    existing_tasks.update(task_info)
    
    with open(task_file, "w", encoding='utf-8') as f:
        json.dump(existing_tasks, f, ensure_ascii=False, indent=2)

def remove_completed_task(task_file, task_id):
    """删除已完成的任务"""
    existing_tasks = load_task_info(task_file)
    if task_id in existing_tasks:
        del existing_tasks[task_id]
        with open(task_file, "w", encoding='utf-8') as f:
            json.dump(existing_tasks, f, ensure_ascii=False, indent=2)
        print(f"已从任务列表中移除: {task_id}")

def generate_3d_asset(prompt, session=None):
    if session is None:
        session = create_session()
        
    headers = {
        "Authorization": f"Bearer {RODIN_API_KEY}",
    }
    
    # 使用multipart/form-data格式
    files = {
        'prompt': (None, prompt),
        'geometry_file_format': (None, 'obj'),
        'quality': (None, 'medium')
    }
    
    response = session.post(
        API_ENDPOINT, 
        headers=headers, 
        files=files,
        timeout=(10, 60)
    )
    
    # 在仅生成模式下减少输出
    response_data = response.json()
    if response.status_code == 201:
        print(f"✅ API调用成功，状态码: {response.status_code}")
    else:
        print("response:")
        print(response_data)
    
    if response.status_code == 201:
        task_id = response_data.get("uuid")  # 正确的字段名是uuid
        print(f"🎯 生成任务已创建，任务ID: {task_id}")
        return task_id
    else:
        print(f"❌ 请求失败，状态码: {response.status_code}")
        print(response.text)
        return None

def check_task_status(task_id, session=None):
    if session is None:
        session = create_session()
        
    ENDPOINT = "https://api.hyper3d.com/api/v2/download"
    
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {RODIN_API_KEY}',
    }

    data = {
        "task_uuid": task_id
    }

    response = session.post(
        ENDPOINT, 
        headers=headers, 
        json=data,
        timeout=(10, 60)
    )
    
    if response.status_code == 201 and response.json().get("list"):
        file_list = response.json().get("list")
        download_files = {}
        
        file_names = []
        for file in file_list:
            file_name = file.get('name')
            file_names.append(file_name)
            download_files[file_name] = file.get('url')
        
        print(f"检测到文件: {', '.join(file_names)}")
        
        # 只有当存在OBJ文件时才认为任务完成
        if len(file_names) > 1:
            return download_files
        else:
            print("⏳ 仅有预览文件，等待3D模型生成...")
    
    return None

def generate_only_mode(prompts, download_dir, session):
    """仅生成模式：只提交任务，不下载"""
    task_file = os.path.join(download_dir, "task_ids.json")
    
    new_task_info = {}
    for i, prompt in enumerate(prompts, 1):
        print(f"\n正在提交第{i}个物体: {prompt}")
        task_id = generate_3d_asset(prompt, session)
        if task_id:
            new_task_info[task_id] = prompt
            print(f"✅ 任务提交成功: {task_id}")
        else:
            print(f"❌ 提交任务失败, prompt: {prompt[:20]}...")

    if new_task_info:
        save_task_info(task_file, new_task_info)
        print(f"\n🎯 本次提交了 {len(new_task_info)} 个任务")
        print(f"📁 任务信息已保存到: {task_file}")
        print("💡 使用 '--mode download' 选项来下载生成的模型")
    else:
        print("❌ 没有成功提交任何任务")

def download_only_mode(download_dir, session):
    """仅下载模式：读取任务文件并下载完成的任务"""
    task_file = os.path.join(download_dir, "task_ids.json")
    task_info = load_task_info(task_file)
    
    if not task_info:
        print("📭 没有找到待下载的任务")
        print(f"请检查文件: {task_file}")
        return
    
    task_ids = list(task_info.keys())
    print(f"📋 找到 {len(task_ids)} 个待下载的任务")
    
    completed_tasks = []
    for task_id in task_ids:
        print(f"\n📋 检查任务状态: {task_id}")
        download_files = check_task_status(task_id, session)
        if download_files:
            print(f"🎉 任务完成，开始下载 {len(download_files)} 个文件...")
            os.makedirs(os.path.join(download_dir, task_id), exist_ok=True)
            
            # 获取任务对应的prompt
            current_prompt = task_info.get(task_id, "未知prompt")
            with open(os.path.join(download_dir, task_id, "info.json"), "w", encoding='utf-8') as f:
                json.dump({
                    "prompt": current_prompt,
                    "task_id": task_id,
                    "download_files": download_files
                }, f, ensure_ascii=False, indent=2)
                
            # 下载所有文件
            for file_name, download_url in download_files.items():
                print(f"⬇️  正在下载: {file_name}")
                response = session.get(download_url, timeout=(10, 120))
                with open(os.path.join(download_dir, task_id, file_name), "wb") as f:
                    f.write(response.content)
                print(f"✅ 下载完成: {file_name}")
            
            completed_tasks.append(task_id)
            print(f"🎊 任务 {task_id} 下载完成！")
        else:
            print(f"⏳ 任务 {task_id} 仍在生成中...")
    
    # 删除已完成的任务
    for task_id in completed_tasks:
        remove_completed_task(task_file, task_id)
    
    if completed_tasks:
        print(f"\n✅ 完成下载 {len(completed_tasks)} 个任务")
    
    remaining_tasks = len(task_ids) - len(completed_tasks)
    if remaining_tasks > 0:
        print(f"⏳ 还有 {remaining_tasks} 个任务仍在生成中，请稍后再次运行下载")

def generate_and_download_mode(prompts, download_dir, session):
    """生成并下载模式：提交任务后等待并下载"""
    task_file = os.path.join(download_dir, "task_ids.json")
    
    # 先提交所有任务
    new_task_info = {}
    for i, prompt in enumerate(prompts, 1):
        print(f"\n正在提交第{i}个物体: {prompt}")
        task_id = generate_3d_asset(prompt, session)
        if task_id:
            new_task_info[task_id] = prompt
        else:
            print(f"提交任务失败, prompt: {prompt[:20]}...")

    if not new_task_info:
        print("❌ 没有成功提交任何任务")
        return
    
    # 保存任务信息
    save_task_info(task_file, new_task_info)
    
    task_ids = list(new_task_info.keys())
    print(f"📋 已提交 {len(task_ids)} 个任务，开始等待生成...")

    # 等待并下载
    while task_ids:
        try:
            time.sleep(20)
            for task_id in task_ids[:]:
                print(f"\n📋 检查任务状态: {task_id}")
                download_files = check_task_status(task_id, session)
                if download_files:
                    print(f"🎉 任务完成，开始下载 {len(download_files)} 个文件...")
                    os.makedirs(os.path.join(download_dir, task_id), exist_ok=True)
                    
                    current_prompt = new_task_info.get(task_id, "未知prompt")
                    with open(os.path.join(download_dir, task_id, "info.json"), "w", encoding='utf-8') as f:
                        json.dump({
                            "prompt": current_prompt,
                            "task_id": task_id,
                            "download_files": download_files
                        }, f, ensure_ascii=False, indent=2)
                        
                    for file_name, download_url in download_files.items():
                        print(f"⬇️  正在下载: {file_name}")
                        response = session.get(download_url, timeout=(10, 120))
                        with open(os.path.join(download_dir, task_id, file_name), "wb") as f:
                            f.write(response.content)
                        print(f"✅ 下载完成: {file_name}")
                    
                    task_ids.remove(task_id)
                    remove_completed_task(task_file, task_id)
                    print(f"🎊 任务 {task_id} 完成下载！")
            
            if not task_ids:
                print("🎉 所有任务已完成!")
                break
                
        except KeyboardInterrupt:
            print("\n⏸️  程序中断")
            print(f"💾 剩余任务已保存到: {task_file}")
            print("💡 使用 '--mode download' 继续下载")
            break

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Rodin 3D模型生成工具")
    parser.add_argument(
        "--mode", 
        choices=["generate", "download", "both"], 
        default="generate",
        help="运行模式: generate(仅生成,默认), download(仅下载), both(生成并下载)"
    )
    
    args = parser.parse_args()
    
    # 检查API密钥（在需要生成时）
    if args.mode in ["generate", "both"] and not RODIN_API_KEY:
        print("❌ 请设置RODIN_API_KEY环境变量")
        print("例如: export RODIN_API_KEY=your_api_key_here")
        exit(1)
    
    if os.path.exists("prompt.txt"):
        with open("prompt.txt", "r", encoding='utf-8') as f:
            prompts = f.readlines()
            prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
    else:
        prompts = [
            "生成一个模型玩具车。高度细节化的科幻装甲战车模型，流线型钛合金车身带有发光能量槽，六轮全地形悬浮底盘，车顶配备可旋转等离子炮台，车体有仿生机械纹理和全息投影仪表盘，整体采用赛博朋克风格的霓虹蓝紫配色，表面有纳米涂层反光效果，背景是火星荒漠场景"
        ]

    download_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    os.makedirs(download_dir, exist_ok=True)
    
    # 创建session
    session = create_session()
    
    print(f"🚀 运行模式: {args.mode}")
    print(f"📁 输出目录: {download_dir}")
    
    if args.mode == "generate":
        print("📝 仅生成模式 - 只提交任务，不下载")
        generate_only_mode(prompts, download_dir, session)
        
    elif args.mode == "download":
        print("⬇️  仅下载模式 - 检查并下载已完成的任务")
        download_only_mode(download_dir, session)
        
    elif args.mode == "both":
        print("🔄 生成并下载模式 - 提交任务后等待下载")
        generate_and_download_mode(prompts, download_dir, session)
