import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 解决图表中负号无法正常显示的问题
plt.rcParams['axes.unicode_minus'] = False  

# 加载当前目录下的本地字体文件
# 如果字体文件不在同一目录，请将 "MSYH.TTC" 替换为绝对路径
try:
    title_font = FontProperties(fname="MSYH.TTC", size=14)
    label_font = FontProperties(fname="MSYH.TTC", size=12)
    legend_font = FontProperties(fname="MSYH.TTC", size=10)
except Exception as e:
    print(f"字体加载失败，请检查 MSYH.TTC 文件是否存在: {e}")

def project_to_simplex(v):
    """
    单纯形投影算子 (Simplex Projection Operator)
    作用：确保防守方更新后的概率 p1, p2, p3 满足 p_i >= 0 且 sum(p_i) = 1
    """
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

def run_game_simulation(epochs=200, eta_D=0.01, eta_A=0.01):
    """
    运行Stackelberg博弈的投影梯度下降仿真
    """
    # ================= 1. 参数初始化 =================
    # 防守方收益参数 B (单位: Mbps，挽回的正常吞吐量)
    B_F = np.array([80, 60, 95])  # 面对泛洪攻击(Flood)时的收益
    B_S = np.array([10, 50, 90])  # 面对隐蔽/拟态攻击(Stealthy)时的收益

    # 防守方成本参数 C (归一化开销: CPU、内存、流表占用)
    C = np.array([5, 20, 70])     

    # 攻击方基础破坏收益 G
    G_F = 100  
    G_S = 80   

    # 防守策略给攻击方带来的惩罚/阻断损失 L
    L_F = np.array([90, 60, 95])  
    L_S = np.array([10, 40, 85])  

    # ================= 2. 策略概率初始化 =================
    p = np.array([0.33, 0.33, 0.34])
    a = 0.5

    history_p1, history_p2, history_p3 = [], [], []
    history_a = []

    # ================= 3. 梯度下降迭代过程 =================
    for t in range(epochs):
        history_p1.append(p[0])
        history_p2.append(p[1])
        history_p3.append(p[2])
        history_a.append(a)

        # 计算防守方和攻击方的梯度
        grad_p = a * B_F + (1 - a) * B_S - C
        grad_a = np.sum(p * ((G_F - G_S) - (L_F - L_S)))

        # 策略更新与投影
        p = project_to_simplex(p + eta_D * grad_p)
        a = np.clip(a + eta_A * grad_a, 0.0, 1.0)

    # ================= 4. 绘制策略收敛曲线 =================
    plt.figure(figsize=(10, 6))
    
    # 绘制防守方策略变化
    plt.plot(history_p1, label='p1: 流表阻断 (Flow-table Drop)', linewidth=2, color='blue')
    plt.plot(history_p2, label='p2: 动态限速 (Dynamic Rate Limiting)', linewidth=2, color='orange')
    plt.plot(history_p3, label='p3: 深度清洗 (VNF Deep Scrubbing)', linewidth=2, color='green')
    
    # 绘制攻击方策略变化 (虚线)
    plt.plot(history_a, label='a: 泛洪攻击概率 (Flood Attack Prob)', linewidth=2, linestyle='--', color='red')

    # 使用本地字体渲染标题和坐标轴
    plt.title('基于投影梯度下降的动态防御策略收敛图', fontproperties=title_font)
    plt.xlabel('决策迭代时间步长 (Decision Iterations)', fontproperties=label_font)
    plt.ylabel('策略分配概率 (Strategy Probability)', fontproperties=label_font)
    
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # 使用本地字体渲染图例
    plt.legend(loc='center right', prop=legend_font)
    plt.tight_layout()
    
    plt.savefig('strategy_convergence.png', dpi=300)
    print("图表已成功保存为 strategy_convergence.png，中文渲染完成。")
    print(f"迭代结束。纳什均衡点：\n防守方策略 p = {p}\n攻击方泛洪概率 a = {a}")

if __name__ == "__main__":
    run_game_simulation(epochs=300, eta_D=0.02, eta_A=0.01)