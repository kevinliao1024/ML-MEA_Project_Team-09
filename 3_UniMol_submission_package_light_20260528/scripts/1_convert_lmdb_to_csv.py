#!/usr/bin/env python3
"""
脚本1: 将官方 LMDB 分子属性数据转换为 unimol_tools 可用的 CSV 格式
用法: python scripts/1_convert_lmdb_to_csv.py
"""

import os
import sys
import pickle
import csv
import gzip
from collections import Counter

# 已跳过的 SMILES 原因统计
_skip_reasons = Counter()

REQUIRED_TASKS = ("bbbp", "esol", "qm9dft", "tox21")
MULTILABEL_TASKS = {"tox21", "toxcast", "sider", "pcba", "muv"}

RAW_TASK_ALIASES = {
    "qm7dft": "qm7",
    "qm8dft": "qm8",
    "qm9dft": "qm9",
}

QM9DFT_TARGET_COLUMNS = ["homo", "lumo", "gap"]

def is_valid_smiles(smiles):
    """用 RDKit 检查 SMILES 是否合法可解析，返回 (True/False, reason)"""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "RDKit 无法解析"
        return True, None
    except Exception as e:
        return False, f"解析异常: {e}"


def _target_values(target):
    if hasattr(target, "tolist"):
        target = target.tolist()
    if isinstance(target, (tuple, list)):
        return list(target)
    return [target]


def _float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _clean_target_value(value, task_name):
    value = _float_or_none(value)
    if task_name in MULTILABEL_TASKS and value == -1.0:
        return float("nan")
    return value


def _raw_task_name(task_name):
    return RAW_TASK_ALIASES.get(task_name, task_name)


def _read_raw_target_columns(task_name, data_dir):
    raw_name = _raw_task_name(task_name)
    raw_dir = os.path.join(str(data_dir), "raw_data")
    for suffix, opener in [(".csv", open), (".csv.gz", gzip.open)]:
        path = os.path.join(raw_dir, raw_name + suffix)
        if not os.path.isfile(path):
            continue
        with opener(path, "rt", newline="") as f:
            header = next(csv.reader(f))
        lowered = [col.lower() for col in header]
        if "smiles" not in lowered:
            return []
        smiles_idx = lowered.index("smiles")
        return [col for col in header[smiles_idx + 1:] if col]
    return []


def resolve_target_columns(task_name, target_count, data_dir):
    if target_count == 1:
        return ["TARGET"]

    raw_columns = _read_raw_target_columns(task_name, data_dir)
    if task_name == "qm9dft" and target_count == 3:
        if all(col in raw_columns for col in QM9DFT_TARGET_COLUMNS):
            return QM9DFT_TARGET_COLUMNS

    if len(raw_columns) == target_count:
        return raw_columns

    return [f"TARGET_{i + 1}" for i in range(target_count)]


def build_record(smiles, target, target_columns, task_name=None):
    values = [_clean_target_value(value, task_name) for value in _target_values(target)]
    if len(values) < len(target_columns):
        values.extend([None] * (len(target_columns) - len(values)))
    elif len(values) > len(target_columns):
        values = values[:len(target_columns)]

    record = {"SMILES": smiles}
    record.update(dict(zip(target_columns, values)))
    return record


def discover_task_names(data_dir):
    """Return only the project-required task directories."""
    return [
        task
        for task in REQUIRED_TASKS
        if os.path.isdir(os.path.join(str(data_dir), task))
    ]

def lmdb_to_csv(lmdb_path, output_csv, task_name):
    """
    读取 LMDB 文件并转换为 CSV
    task_name: 任务名称（决定 target 列名）
    """
    import lmdb
    import pandas as pd

    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    cursor = txn.cursor()
    keys = list(cursor.iternext(values=False))

    print(f"  [{task_name}] 共 {len(keys)} 条数据，开始转换...")

    target_columns = None
    records = []
    for i, key in enumerate(keys):
        raw = txn.get(key)
        try:
            data = pickle.loads(raw)
        except Exception as e:
            print(f"  跳过无法解析的记录 {key}: {e}")
            continue

        # 官方数据统一结构：
        #  - smiles: SMILES 字符串
        #  - target: 目标值（可能是标量或列表）
        smiles = data.get("smiles", data.get("SMILES", ""))
        target = data.get("target", data.get("TARGET", None))

        if not smiles or target is None:
            # 尝试其他可能的键
            for k in data:
                if "smi" in k.lower():
                    smiles = data[k]
                if "targ" in k.lower() or k.lower() in ("y", "label", "labels"):
                    target = data[k]

        target_values = _target_values(target)
        if target_columns is None:
            data_dir = os.path.dirname(os.path.dirname(lmdb_path))
            target_columns = resolve_target_columns(task_name, len(target_values), data_dir)

        # RDKit SMILES 合法性检查
        valid, reason = is_valid_smiles(smiles)
        if not valid:
            _skip_reasons[reason] += 1
            continue

        records.append(build_record(smiles, target_values, target_columns, task_name))

        if (i + 1) % 5000 == 0:
            print(f"  已处理 {i + 1}/{len(keys)} ...")

    env.close()

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"  ✅ 已保存: {output_csv} ({len(df)} 条)")


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, "data", "molecular_property_prediction")
    out_dir = os.path.join(base, "csv_data")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(data_dir):
        print(f"❌ 未找到数据目录: {data_dir}")
        print("请先将 molecular_property_prediction.tar.gz 解压到 data/ 目录")
        sys.exit(1)

    # 只转换当前项目要求的四个任务
    subdirs = discover_task_names(data_dir)

    print(f"将转换 {len(subdirs)} 个任务子目录: {subdirs}")
    print("-" * 60)

    for task_name in subdirs:
        task_path = os.path.join(data_dir, task_name)
        # 每个子目录有 train.lmdb / valid.lmdb / test.lmdb
        for split in ["train", "valid", "test"]:
            lmdb_file = os.path.join(task_path, f"{split}.lmdb")
            if not os.path.isfile(lmdb_file):
                continue
            out_csv = os.path.join(out_dir, f"{task_name}_{split}.csv")
            print(f"[{task_name}] 转换 {split}.lmdb -> {os.path.basename(out_csv)}")
            try:
                lmdb_to_csv(lmdb_file, out_csv, task_name)
            except Exception as e:
                print(f"  ❌ 转换失败: {e}")
            print()

    print("=" * 60)
    print(f"✅ 全部转换完成！CSV 文件保存在: {out_dir}")
    if _skip_reasons:
        print(f"\n⚠️  跳过 {sum(_skip_reasons.values())} 条不合法 SMILES（原因分布）:")
        for reason, count in _skip_reasons.most_common():
            print(f"  - {reason}: {count} 条")


if __name__ == "__main__":
    main()
