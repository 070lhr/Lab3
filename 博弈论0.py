import numpy as np
import matplotlib.pyplot as plt

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
    :param epochs: 迭代周期数
    :param eta_D: 防守方学习率 (边缘网关)
    :param eta_A: 攻击方学习率 (僵尸网络)
    """
    # ================= 1. 参数初始化 (依据网络环境设定) =================
    # 防守方收益参数 B (单位: Mbps，挽回的正常吞吐量)
    # [流表级阻断, 动态限速, 深度清洗]
    B_F = np.array([80, 60, 95])  # 面对泛洪攻击(Flood)时的收益
    B_S = np.array([10, 50, 90])  # 面对隐蔽/拟态攻击(Stealthy)时的收益

    # 防守方成本参数 C (归一化开销: CPU、内存、流表占用)
    # 成本关系: 深度清洗(C3) >> 动态限速(C2) > 流表阻断(C1)
    C = np.array([5, 20, 70])     

    # 攻击方基础破坏收益 G
    G_F = 100  # 泛洪攻击的破坏力
    G_S = 80   # 隐蔽拟态攻击的破坏力

    # 防守策略给攻击方带来的惩罚/阻断损失 L
    L_F = np.array([90, 60, 95])  # 对应三种策略防泛洪的压制力
    L_S = np.array([10, 40, 85])  # 对应三种策略防拟态的压制力

    # ================= 2. 策略概率初始化 =================
    # 防守方初始策略 P = [p1, p2, p3]，平均分配
    p = np.array([0.33, 0.33, 0.34])
    # 攻击方初始策略 a (采用泛洪攻击的概率)
    a = 0.5

    # 用于记录绘图数据的列表
    history_p1, history_p2, history_p3 = [], [], []
    history_a = []

    # ================= 3. 梯度下降迭代过程 =================
    for t in range(epochs):
        # 记录当前状态
        history_p1.append(p[0])
        history_p2.append(p[1])
        history_p3.append(p[2])
        history_a.append(a)

        # 3.1 计算防守方效用函数关于 p_i 的梯度 (对应公式 5-1 的导数)
        # grad_p_i = a * B_i^F + (1-a) * B_i^S - C_i
        grad_p = a * B_F + (1 - a) * B_S - C

        # 3.2 计算攻击方效用函数关于 a 的梯度 (对应公式 5-4)
        grad_a = np.sum(p * ((G_F - G_S) - (L_F - L_S)))

        # 3.3 策略更新与投影 (对应公式 5-7 和 5-8)
        # 防守方沿着梯度方向增加收益，并映射回有效概率空间
        p = project_to_simplex(p + eta_D * grad_p)
        
        # 攻击方同样沿着梯度方向更新，概率限制在 [0, 1] 之间
        a = np.clip(a + eta_A * grad_a, 0.0, 1.0)

    # ================= 4. 绘制策略收敛曲线 =================
    plt.figure(figsize=(10, 6))
    
    # 绘制防守方策略变化
    plt.plot(history_p1, label='p1: Flow-table Drop (流表阻断)', linewidth=2, color='blue')
    plt.plot(history_p2, label='p2: Dynamic Rate Limiting (动态限速)', linewidth=2, color='orange')
    plt.plot(history_p3, label='p3: VNF Deep Scrubbing (深度清洗)', linewidth=2, color='green')
    
    # 绘制攻击方策略变化 (虚线)
    plt.plot(history_a, label='a: Flood Attack Prob (泛洪攻击概率)', linewidth=2, linestyle='--', color='red')

    plt.title('Convergence of Dynamic Defense Strategies via Projected Gradient Descent', fontsize=14)
    plt.xlabel('Decision Iterations (时间步长)', fontsize=12)
    plt.ylabel('Strategy Probability (策略概率)', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='center right', fontsize=10)
    plt.tight_layout()
    
    # 保存高清图表用于论文插入
    plt.savefig('strategy_convergence.png', dpi=300)
    plt.show()

    print(f"迭代结束。纳什均衡点：\n防守方策略 p = {p}\n攻击方泛洪概率 a = {a}")

if __name__ == "__main__":
    run_game_simulation(epochs=300, eta_D=0.02, eta_A=0.01)