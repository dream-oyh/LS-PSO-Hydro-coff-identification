import matplotlib.pyplot as plt


def plot_velocity_comparison(
    timestamps, true_linear_vel, true_angular_vel, pred_linear_vel, pred_angular_vel
):
    """
    绘制真实速度和预测速度的对比图

    参数:
    timestamps: 时间序列
    true_linear_vel: 真实线速度 (nx3)
    true_angular_vel: 真实角速度 (nx3)
    pred_linear_vel: 预测线速度 (nx3)
    pred_angular_vel: 预测角速度 (nx3)
    """
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    fig.suptitle("Velocity Comparison (True vs Predicted)", fontsize=16)

    # 线速度标签
    linear_labels = ["Surge (X)", "Sway (Y)", "Heave (Z)"]
    # 角速度标签
    angular_labels = ["Roll (ϕ)", "Pitch (θ)", "Yaw (ψ)"]

    # 绘制线速度对比
    for i in range(3):
        ax = axes[0, i]
        ax.plot(timestamps, true_linear_vel[:, i], "b-", label="True", alpha=0.7)
        ax.plot(timestamps, pred_linear_vel[:, i], "r--", label="Predicted", alpha=0.7)
        ax.set_title(f"Linear Velocity - {linear_labels[i]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.grid(True)
        ax.legend()

    # 绘制角速度对比
    for i in range(3):
        ax = axes[1, i]
        ax.plot(timestamps, true_angular_vel[:, i], "b-", label="True", alpha=0.7)
        ax.plot(timestamps, pred_angular_vel[:, i], "r--", label="Predicted", alpha=0.7)
        ax.set_title(f"Angular Velocity - {angular_labels[i]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angular Velocity (rad/s)")
        ax.grid(True)
        ax.legend()

    # 调整子图布局
    plt.tight_layout()
    plt.show()
