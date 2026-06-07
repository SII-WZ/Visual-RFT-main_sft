import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

def plot_training_convergence(root_dir: str, save_path: str = "training_convergence.png", 
                              metrics: list = None, figsize: tuple = (12, 8), 
                              dpi: int = 300, font_size: int = 10):
    """
    批量读取目标目录下所有的.json文件，提取训练指标并绘制收敛图，保存到本地。
    
    Args:
        root_dir (str): 包含所有.json指标文件的根目录路径（如 lisa_evaluation/2b_eval）
        save_path (str): 收敛图的保存路径（含文件名和格式，支持png/jpg/pdf等）
        metrics (list): 需要绘制的指标列表，默认提取["kl", "loss", "reward", "reward_std"]
        figsize (tuple): 图像尺寸，格式为(宽度, 高度)
        dpi (int): 图像分辨率，dpi越高图像越清晰
        font_size (int): 图像字体大小
    """
    # 1. 设置默认参数和初始化数据存储
    if metrics is None:
        metrics = ["kl", "loss", "reward", "reward_std"]
    
    # 用于存储所有提取的指标数据，key为指标名，value为(step列表, 指标值列表)
    all_metrics_data = {metric: ([], []) for metric in metrics}
    # 用于去重（避免不同json文件重复读取同一step的数据）
    recorded_steps = set()

    # 2. 批量查找目标目录下所有的.json文件（核心修改：不再匹配checkpoint文件夹）
    json_pattern = os.path.join(root_dir, "*.json")  # 直接匹配根目录下所有.json文件
    json_files = glob.glob(json_pattern)

    if not json_files:
        raise FileNotFoundError(f"在根目录 {root_dir} 下未找到任何.json文件")

    # 3. 遍历每个.json文件，提取数据
    for file_path in json_files:
        # 跳过可能的隐藏json文件（如 .DS_Store.json，可选）
        file_name = os.path.basename(file_path)
        if file_name.startswith("."):
            continue
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            
            # 兼容两种常见json格式：① 直接包含log_history ② 顶层即为单条日志数据
            log_history = []
            # 情况1：json包含log_history字段（与原trainer_state.json格式一致）
            if "log_history" in json_data:
                log_history = json_data.get("log_history", [])
            # 情况2：json本身就是单条日志，或直接是日志列表（适配grpo_metrics_*.json）
            else:
                if isinstance(json_data, list):
                    log_history = json_data
                else:
                    log_history = [json_data]
            
            if not log_history:
                print(f"警告：{file_path} 中未找到有效日志数据，跳过该文件")
                continue
            
            # 遍历每条日志，提取指定指标
            for log_entry in log_history:
                step = log_entry.get("step")
                # 若日志中无step，用文件名中的数字（如50、100）作为step（兼容grpo_metrics_*.json）
                if step is None:
                    # 从文件名中提取数字作为step（如 grpo_metrics_50.json → 50）
                    import re
                    num_match = re.search(r'(\d+)', file_name)
                    if num_match:
                        step = int(num_match.group(1))
                    else:
                        continue  # 无有效step，跳过该条日志
                
                if step in recorded_steps:
                    continue  # 跳过已记录的step，避免重复
                
                # 提取当前step的所有指定指标（支持嵌套指标如 rewards/format_reward）
                for metric in metrics:
                    # 处理嵌套指标（如 "rewards/format_reward"）
                    if "/" in metric:
                        keys = metric.split("/")
                        metric_value = json_data
                        try:
                            for key in keys:
                                metric_value = metric_value.get(key)
                                if metric_value is None:
                                    break
                        except:
                            metric_value = None
                    else:
                        metric_value = log_entry.get(metric)
                    
                    if metric_value is not None:  # 仅提取存在的指标值
                        all_metrics_data[metric][0].append(step)
                        all_metrics_data[metric][1].append(metric_value)
                
                # 标记该step已记录
                recorded_steps.add(step)
        
        except Exception as e:
            print(f"警告：处理文件 {file_path} 时出错，错误信息：{str(e)}，跳过该文件")
            continue

    # 4. 数据预处理：按step排序（保证曲线顺序正确）
    for metric in metrics:
        steps, values = all_metrics_data[metric]
        if len(steps) > 0:
            # 按step升序排序
            sorted_pairs = sorted(zip(steps, values), key=lambda x: x[0])
            all_metrics_data[metric] = ([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs])
        else:
            print(f"警告：指标 {metric} 未提取到任何有效数据，绘制时将跳过")

    # 5. 绘制收敛图
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 支持中文标签（Windows）
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 定义颜色列表（支持更多指标扩展）
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    
    for idx, metric in enumerate(metrics):
        steps, values = all_metrics_data[metric]
        if len(steps) == 0:
            continue
        
        # 绘制指标曲线（带标记点，便于查看离散数据）
        ax.plot(steps, values, color=colors[idx % len(colors)], label=metric, 
                linewidth=2, marker=".", markersize=4, alpha=0.8)

    # 6. 设置图像标签和样式
    ax.set_xlabel("Training Step", fontsize=font_size + 2)
    ax.set_ylabel("Metric Value", fontsize=font_size + 2)
    ax.set_title("Training Convergence Curve of Key Metrics", fontsize=font_size + 4, pad=20)
    ax.grid(True, alpha=0.3, linestyle="--")  # 添加网格线，便于读取
    ax.legend(fontsize=font_size, loc="best")  # 自动选择最佳图例位置
    plt.tight_layout()  # 自动调整布局，避免标签重叠

    # 7. 保存图像到本地（核心修复：原save_path缺少文件名，自动补充默认文件名）
    # 若传入的save_path是目录，自动补充默认文件名
    if os.path.isdir(save_path):
        save_path = os.path.join(save_path, "training_convergence.png")
    
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 若保存目录不存在，创建目录
    
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"收敛图已成功保存到：{os.path.abspath(save_path)}")


if __name__ == "__main__":
    # 替换为你的.json文件所在根目录（包含 grpo_metrics_50.json、grpo_metrics_100.json 等）
    ROOT_DIR = "/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/lisa_evaluation/2b_eval"
    
    # 调用函数绘制收敛图
    plot_training_convergence(
        root_dir=ROOT_DIR,
        # 修复：原save_path是目录，现在可直接传入目录（函数会自动补充文件名），也可指定具体文件名
        save_path="/inspire/hdd/global_user/zhouwei-240108540164/autodrive/Visual-RFT-main/lisa_evaluation/7b_eval",
        # 支持你的目标指标（包括嵌套指标如 rewards/format_reward）
        metrics=["average_kl"], #"average_accuracy_reward", "average_format_reward" "average_loss" "average_kl"
        figsize=(14, 9),  # 图像尺寸
        dpi=300  # 高分辨率，适合论文使用
    )
    
#  "average_loss": 0.021493742299978347,
#     "average_kl": 0.2149088882622035,
#     "average_reward": 2.351559627406737,
#     "average_reward_std": 0.0,
#     "average_accuracy_reward": 1.3515596234184855,
#     "average_format_reward": 1.0,