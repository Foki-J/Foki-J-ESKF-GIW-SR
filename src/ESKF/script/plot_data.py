import matplotlib.pyplot as plt

# 解析数据
data = """
d1:0.00014954  -0.300025   0.240006 r:0.41
d1:0.000149896   -0.300024    0.239994 r:0.41
d1:0.000148765   -0.300027    0.239998 r:0.41
d1:0.000147668   -0.300023    0.239996 r:0.41
d1:0.000132786   -0.300007    0.239985 r:0.410008
d1:0.000149715   -0.300021    0.240006 r:0.409996
d1:0.000446622   -0.300043    0.241264 r:0.410468
d1:0.00103434  -0.300328   0.242685 r:0.421483
d1:0.000471644   -0.300655    0.239677 r:0.431059
d1:0.000723964   -0.301179    0.240939 r:0.43475
d1:0.00155331  -0.300508   0.242406 r:0.436892
d1:0.00163145  -0.299976   0.241932 r:0.437254
d1:0.0017174 -0.298593  0.243381 r:0.437406
d1:0.00277187  -0.297966   0.243514 r:0.437733
d1:0.00559693  -0.298385   0.247655 r:0.438056
d1:0.0061561 -0.298217   0.24827 r:0.43829
d1:0.00632236  -0.298124   0.248287 r:0.438148
d1:0.00611521  -0.298391   0.247893 r:0.438195
d1:0.00601307  -0.298574   0.247677 r:0.438754
d1:0.0061072 -0.298655  0.247923 r:0.438419
d1:0.00593555  -0.291555   0.247485 r:0.436959
"""

# 初始化列表存储数据
seconds = []
d1_values = []
r_values = []

# 解析每一行数据
lines = data.strip().split('\n')
for line in lines:
    parts = line.split()
    d1_value = float(parts[2])
    r_value = float(parts[-1][2:])
    seconds.append(len(seconds) * 1)  # 假设每条记录间隔1秒
    d1_values.append(d1_value)
    r_values.append(r_value)

# 创建图形
fig, ax1 = plt.subplots()

# 绘制d1值
color = 'tab:red'
ax1.set_xlabel('Seconds')
ax1.set_ylabel('d1', color=color)
ax1.plot(seconds, d1_values, color=color, label='d1')
ax1.tick_params(axis='y', labelcolor=color)

# 创建第二个y轴来绘制r值
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('r', color=color)
ax2.plot(seconds, r_values, color=color, label='r')
ax2.tick_params(axis='y', labelcolor=color)

# 添加图例
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

# 设置标题
plt.title('d1 and r over Time')

# 显示图形
plt.show()