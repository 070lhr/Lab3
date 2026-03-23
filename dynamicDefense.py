import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.patches as patches

# ================= 1. 环境配置 =================
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams['mathtext.fontset'] = 'stix'
try:
    font = FontProperties(fname="MSYH.TTC", size=11)
    title_font = FontProperties(fname="MSYH.TTC", size=14)
except:
    font = title_font = None

# ================= 2. 底层物联网节点类定义 =================
class IoTNode:
    def __init__(self, node_id, x, y, node_type):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.node_type = node_type  # 'benign'(合法), 'attacker'(DDoS攻击者), 'selfish'(自私节点)
        self.trust_value = 1.0      # 初始信任值最高为 1.0
        self.is_isolated = False    # 是否被网关切断连接
        
    def generate_traffic(self):
        """模拟单个节点在当前轮次产生的数据量 (bits)"""
        if self.is_isolated:
            return 0
            
        if self.node_type == 'benign':
            return np.random.poisson(200) # 正常节点每次产生约 200 bits
        elif self.node_type == 'attacker':
            return np.random.poisson(85000) # 攻击节点疯狂洪泛
        elif self.node_type == 'selfish':
            # 自私节点：偶尔发大量数据抢占信道，但不完全像 DDoS
            return np.random.poisson(4000) if np.random.rand() > 0.5 else 0

    def update_trust(self, penalty):
        """更新信任值"""
        self.trust_value = max(0.0, self.trust_value - penalty)

# ================= 3. 边缘网关与博弈防御类 =================
class EdgeGateway:
    def __init__(self):
        # 防守策略概率: [流表阻断, 动态限速, 深度清洗]
        self.p = np.array([0.33, 0.33, 0.34])
        # Saiyed 论文中提到的博弈论成本二次函数预设参数
        self.eta_T = -3e-2
        self.gamma_T = 4.0

    def project_to_simplex(self, v):
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        return np.maximum(v - theta, 0)

    def calculate_game_theory_defense(self, total_traffic):
        """执行博弈论调度计算"""
        # 这里的计算将底层拓扑传上来的总流量代入效用函数
        C = np.array([5.0, 15.0, 20.0 + total_traffic * 0.001])
        B_F = np.array([total_traffic * 0.9, total_traffic * 0.5, total_traffic * 0.8])
        B_S = np.array([10, 80, 120])
        
        # 梯度更新 (简化版)
        a = 0.5  # 假设攻击方的某种隐蔽概率
        grad_p = a * B_F + (1 - a) * B_S - C
        self.p = self.project_to_simplex(self.p + 0.05 * grad_p)
        
        # 返回当前最优策略的索引 (0, 1, 或 2)
        return np.argmax(self.p)

# ================= 4. 空间拓扑与离散事件仿真主循环 =================
def run_topological_simulation():
    # --- 参数设定 (完全对标论文) ---
    AREA_SIZE = 200        # 200 x 200 平方米
    NUM_NODES = 100        # 100 个节点
    COMM_RADIUS = 20       # 通信半径 (设为20米以便于网关覆盖，原设2米易造成网络孤岛)
    LINK_CAPACITY = 800000 # 链路总容量 C
    ROUNDS = 100           # 实验总轮次 T_n
    
    # --- 实例化网络节点 ---
    nodes = []
    for i in range(NUM_NODES):
        x, y = np.random.uniform(0, AREA_SIZE, 2)
        # 设定网络中有 70个正常节点, 20个DDoS攻击者, 10个自私节点
        if i < 70: n_type = 'benign'
        elif i < 90: n_type = 'attacker'
        else: n_type = 'selfish'
        nodes.append(IoTNode(i, x, y, n_type))
        
    gateway = EdgeGateway()
    
    # 数据记录
    history_avg_trust = []
    history_gateway_load = []
    history_strategy = []
    
    print(f"开始 {ROUNDS} 轮的离散事件拓扑仿真...")
    
    for round_idx in range(ROUNDS):
        total_traffic = 0
        node_traffic_dict = {}
        
        # 1. 数据收集阶段
        for node in nodes:
            traffic = node.generate_traffic()
            total_traffic += traffic
            node_traffic_dict[node.node_id] = traffic
            
        # 2. 网关博弈论防御决策阶段
        best_strategy = gateway.calculate_game_theory_defense(total_traffic)
        history_strategy.append(best_strategy)
        
        # 3. 动态节点行为与信任惩罚阶段
        active_trusts = [n.trust_value for n in nodes if not n.is_isolated]
        avg_trust = np.mean(active_trusts) if active_trusts else 0
        history_avg_trust.append(avg_trust)
        
        for node in nodes:
            if node.is_isolated: continue
            
            # 引入二次函数成本计算惩罚 (利用 eta_T 和 gamma_T)
            # 流量异常大的节点将受到严厉惩罚
            deviation = node_traffic_dict[node.node_id] - 200
            if deviation > 1000:
                penalty = abs(gateway.eta_T * (deviation ** 2) + gateway.gamma_T * deviation) * 1e-6
                node.update_trust(penalty)
                
            # 【核心机制】信任值低于网络平均水平，直接降级隔离
            if node.trust_value < avg_trust * 0.8:
                node.is_isolated = True
                
        # 记录链路负载率
        load_ratio = min(total_traffic / LINK_CAPACITY, 1.0)
        history_gateway_load.append(load_ratio * 100)

    print("仿真完成！正在生成拓扑与态势图表...")

    # ================= 5. 可视化输出 =================
    fig = plt.figure(figsize=(15, 5))
    
    # 子图1：物理空间拓扑与节点状态图
    ax1 = fig.add_subplot(131)
    ax1.set_xlim(0, AREA_SIZE)
    ax1.set_ylim(0, AREA_SIZE)
    ax1.set_title(f'100节点物联网空间拓扑状态 (第 {ROUNDS} 轮)', fontproperties=title_font)
    
    for node in nodes:
        color = 'green' if node.node_type == 'benign' else ('red' if node.node_type == 'attacker' else 'orange')
        marker = 'x' if node.is_isolated else 'o'
        alpha = 0.3 if node.is_isolated else 1.0
        size = 20 if node.is_isolated else 50
        ax1.scatter(node.x, node.y, c=color, marker=marker, s=size, alpha=alpha)
        
    ax1.text(10, 190, 'Green: Benign (合法)\nRed: Attacker (攻击)\nOrange: Selfish (自私)\nX: Isolated (被隔离)', 
             bbox=dict(facecolor='white', alpha=0.8), fontsize=8)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 子图2：全网平均信任值衰减与动态降级过程
    ax2 = fig.add_subplot(132)
    ax2.plot(range(ROUNDS), history_avg_trust, color='purple', linewidth=2)
    ax2.set_title('动态节点行为：全网平均信任值演变', fontproperties=title_font)
    ax2.set_xlabel('仿真轮次 ($T_n$)', fontproperties=font)
    ax2.set_ylabel('平均信任值 (Trust Value)', fontproperties=font)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 子图3：网关链路负载率与防御动作反馈
    ax3 = fig.add_subplot(133)
    ax3.plot(range(ROUNDS), history_gateway_load, color='darkred', linewidth=2, label='链路负载率(%)')
    ax3.set_title('防御介入后的网关负载演变', fontproperties=title_font)
    ax3.set_xlabel('仿真轮次 ($T_n$)', fontproperties=font)
    ax3.set_ylabel('负载占用百分比 (%)', fontproperties=font)
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('fig_topological_simulation.png', dpi=600, bbox_inches='tight')
    print("图表渲染完成：fig_topological_simulation.png")

if __name__ == "__main__":
    run_topological_simulation()