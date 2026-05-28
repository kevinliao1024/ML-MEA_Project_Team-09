#!/usr/bin/env python3
"""
脚本3: 批量训练所有分子属性预测任务
用法: python scripts/3_train_all.py
"""

import os
import sys
import time

# 按顺序训练所有任务（每个任务用论文中的默认超参数）
TASKS = [
    # 回归任务
    "qm9dft", "qm7dft", "qm8dft",
    "esol", "freesolv", "lipo",
    # 分类任务
    "bbbp", "bace", "clintox", "hiv",
    # 多标签分类
    "tox21", "toxcast", "sider", "pcba", "muv",
]


def main():
    print("=" * 60)
    print("Uni-Mol 分子属性预测 — 批量训练")
    print("=" * 60)
    print(f"共 {len(TASKS)} 个任务: {TASKS}\n")

    from unimol_tools import MolTrain  # 提前检查

    for i, task in enumerate(TASKS, 1):
        print("\n" + "#" * 60)
        print(f"[{i}/{len(TASKS)}] 训练任务: {task}")
        print("#" * 60)

        start = time.time()
        try:
            # 调用 2_train.py 的逻辑（直接复用）
            import subprocess
            result = subprocess.run(
                [sys.executable, "scripts/2_train.py", "--task", task],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            if result.returncode != 0:
                print(f"⚠️  任务 {task} 训练失败，跳过")
        except Exception as e:
            print(f"⚠️  任务 {task} 异常: {e}，跳过")

        elapsed = time.time() - start
        print(f"[{i}/{len(TASKS)}] 完成！耗时: {elapsed//60:.0f}分{elapsed%60:.0f}秒")

    print("\n" + "=" * 60)
    print("全部任务训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
