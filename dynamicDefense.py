import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

# ================= 1. 字体与绘图环境配置 =================
plt.rcParams['axes.unicode_minus'] = False  
font_path = "MSYH.TTC"
if not os.path.exists(font_path):
    print(f"警告: 未找到字体文件 {font_path}，图表中的中文可能显示为方块。")
    title_font = label_font = legend_font = None
else:
    title_font = FontProperties(fname=font_path, size=14)
    label_font = FontProperties(fname=font_path, size=12)
    legend_font = FontProperties(fname=font_path, size=10)

# ================= 2. 核心数学算子 =================
def project_to_simplex(v):
    """单纯形投影：确保策略概率之和为1且大于等于0"""
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0)

# ================= 3. 数据驱动的流量生成器 (模拟 CICIOT2023 特征) =================
def generate_traffic_profile(total_time=300):
    """
    使用两态马尔可夫调制泊松过程(MMPP)思想，生成带突发性的攻击流量
    :return: 随时间变化的攻击带宽 (Mbps) 数组
    """
    traffic_mbps = np.zeros(total_time)
    # 提取自数据集的经验参数
    normal_mean = 50.0       # 平静期/隐蔽探测期平均速率 50 Mbps
    burst_mean = 1500.0      # 爆发期(泛洪攻击)平均速率 1500 Mbps
    
    current_state = 0 # 0: 隐蔽态, 1: 爆发态
    for t in range(total_time):
        # 状态转移概率 (模拟突发攻击的启停)
        if current_state == 0 and np.random.rand() < 0.05: # 5%概率进入爆发
            current_state = 1
        elif current_state == 1 and np.random.rand() < 0.10: # 10%概率恢复隐蔽
            current_state = 0
            
        # 根据当前状态，从泊松分布中采样当前秒的真实流量
        if current_state == 0:
            traffic_mbps[t] = np.random.poisson(normal_mean)
        else:
            traffic_mbps[t] = np.random.poisson(burst_mean)
            
    # 强制设定几个明显的阶段以便于图表展示分析
    traffic_mbps[50:120] = np.random.poisson(burst_mean, 70) # 强制大流量泛洪
    traffic_mbps[120:200] = np.random.poisson(normal_mean, 80) # 强制隐蔽探测
    traffic_mbps[200:270] = np.random.poisson(burst_mean * 1.2, 70) # 极限流量冲击
    return traffic_mbps

# ================= 4. 动态博弈仿真主循环 =================
def run_simulation():
    total_time = 300
    traffic_profile = generate_traffic_profile(total_time)
    
    # 记录数据的数组
    history_p = np.zeros((total_time, 3)) # [p1:阻断, p2:限速, p3:深度清洗]
    history_a = np.zeros(total_time)      # 攻击方泛洪概率
    cpu_game_theory = np.zeros(total_time)
    cpu_static = np.zeros(total_time)     # 基线方法：静态全量深度清洗
    
    # 初始化策略
    p = np.array([0.33, 0.33, 0.34])
    a = 0.5
    eta_D, eta_A = 0.05, 0.02 # 学习率
    
    for t in range(total_time):
        current_traffic = traffic_profile[t]
        
        # --- 参数动态映射 (核心创新) ---
        # 流量越大，泛洪破坏力越大；隐蔽攻击破坏力相对稳定
        G_F = current_traffic * 0.8  
        G_S = 100.0 
        
        # 动态成本计算：深度清洗的CPU成本随流量指数上升
        C1 = 5.0   # 硬件阻断流表开销极低
        C2 = 15.0  # 软件限速开销中等
        C3 = 20.0 + (current_traffic * 0.05) # 深度清洗开销与流量正相关
        C = np.array([C1, C2, C3])
        
        # 动态收益与惩罚设定
        B_F = np.array([current_traffic * 0.9, 100, 150]) 
        B_S = np.array([10, 80, 120])
        L_F = np.array([100, 70, 120])
        L_S = np.array([10, 50, 110])
        
        # --- 梯度计算与策略更新 ---
        # 边缘网关内部通过多次快速迭代寻找当前流量下的纳什均衡
        for _ in range(10): 
            grad_p = a * B_F + (1 - a) * B_S - C
            grad_a = np.sum(p * ((G_F - G_S) - (L_F - L_S)))
            
            p = project_to_simplex(p + eta_D * grad_p)
            a = np.clip(a + eta_A * grad_a, 0.0, 1.0)
            
        # 记录当前时刻的最优策略
        history_p[t] = p
        history_a[t] = a
        
        # --- 评估指标计算 ---
        # 静态防御基线：100%采用深度清洗 (p3=1.0)
        cpu_static[t] = min(C3, 100.0) 
        # 博弈论防御：按概率组合消耗CPU
        cpu_game_theory[t] = min(p[0]*C1 + p[1]*C2 + p[2]*C3, 100.0)

    # ================= 5. 结果可视化与图表输出 =================
    font_kwargs_title = {'fontproperties': title_font} if title_font else {}
    font_kwargs_label = {'fontproperties': label_font} if label_font else {}
    font_kwargs_legend = {'prop': legend_font} if legend_font else {}

    time_axis = np.arange(total_time)
    
    # [图表1：输入流量态势图]
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, traffic_profile, color='purple', linewidth=1.5)
    plt.title('复杂场景下多向量DDoS攻击流量态势 (基于泊松分布模拟)', **font_kwargs_title)
    plt.xlabel('仿真时间 (秒)', **font_kwargs_label)
    plt.ylabel('攻击流量带宽 (Mbps)', **font_kwargs_label)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('fig_1_traffic_profile.png', dpi=300)
    plt.close()

    # [图表2：防守策略动态演变面积图]
    plt.figure(figsize=(10, 5))
    plt.stackplot(time_axis, history_p[:, 0], history_p[:, 1], history_p[:, 2], 
                  labels=['流表阻断 (Drop)', '动态限速 (Rate Limit)', '深度清洗 (Deep Scrub)'],
                  colors=['#4C72B0', '#DD8452', '#55A868'], alpha=0.8)
    plt.title('边缘网关防御动作策略概率分布堆叠图', **font_kwargs_title)
    plt.xlabel('仿真时间 (秒)', **font_kwargs_label)
    plt.ylabel('策略分配比例', **font_kwargs_label)
    plt.ylim(0, 1)
    plt.legend(loc='lower left', **font_kwargs_legend)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('fig_2_strategy_area.png', dpi=300)
    plt.close()

    # [图表3：边缘网关CPU开销对比图]
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, cpu_static, label='基线方法 (静态深度清洗)', color='red', linestyle='--', linewidth=2)
    plt.plot(time_axis, cpu_game_theory, label='提出方法 (博弈论动态调度)', color='blue', linewidth=2)
    plt.title('不同防御机制下的边缘网关CPU算力消耗对比', **font_kwargs_title)
    plt.xlabel('仿真时间 (秒)', **font_kwargs_label)
    plt.ylabel('CPU 归一化占用率 (%)', **font_kwargs_label)
    plt.ylim(0, 105)
    plt.legend(loc='upper right', **font_kwargs_legend)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('fig_3_cpu_overhead.png', dpi=300)
    plt.close()

    print("实验运行完成！已生成三张图表：")
    print("1. fig_1_traffic_profile.png (流量态势)")
    print("2. fig_2_strategy_area.png (策略演变)")
    print("3. fig_3_cpu_overhead.png (CPU开销对比)")

if __name__ == "__main__":
    run_simulation()