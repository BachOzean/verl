#!/usr/bin/env python3
"""
查看 Parquet 文件中指定索引的一条完整记录。

用法示例：
  python3 view_parquet_record.py /path/to/file.parquet -i 0
  python3 view_parquet_record.py /path/to/file.parquet -i -1 --meta
"""

from __future__ import annotations

import argparse
import base64
import datetime
import decimal
import json
import sys
from typing import Any, Dict

try:
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover
    print(
        "未能导入 pyarrow，请先安装：pip install pyarrow",
        file=sys.stderr,
    )
    raise


def json_default_serializer(obj: Any) -> Any:
    """将不可直接 JSON 序列化的对象转换为可序列化形式。"""
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        # 尝试以 int/float 输出；若失败则回退为字符串
        try:
            # 若是整数小数位（如 1.0），优先用 int 更美观
            if obj == obj.to_integral():
                return int(obj)
        except Exception:
            pass
        try:
            return float(obj)
        except Exception:
            return str(obj)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        # 优先尝试按 UTF-8 文本输出，否则用 base64
        raw_bytes = bytes(obj)
        try:
            return raw_bytes.decode("utf-8")
        except Exception:
            return base64.b64encode(raw_bytes).decode("ascii")
    return str(obj)


def get_total_rows(parquet_file: pq.ParquetFile) -> int:
    meta = parquet_file.metadata
    return meta.num_rows if meta is not None else sum(
        parquet_file.metadata.row_group(i).num_rows  # type: ignore[union-attr]
        for i in range(parquet_file.metadata.num_row_groups)  # type: ignore[union-attr]
    )


def read_record_by_index(file_path: str, index: int) -> Dict[str, Any]:
    """在不读取整文件入内存的前提下，按行组定位并读取单条记录。"""
    pf = pq.ParquetFile(file_path)

    if pf.metadata is None or pf.metadata.num_row_groups == 0:
        # 没有行组或无元信息时，直接整体读取（极少见）
        table = pf.read()
        total_rows = table.num_rows
        if total_rows == 0:
            raise ValueError("文件没有任何记录")
        if index < 0:
            index = total_rows + index
        if index < 0 or index >= total_rows:
            raise IndexError(f"索引超出范围：{index}，总行数：{total_rows}")
        return table.slice(index, 1).to_pylist()[0]

    total_rows = pf.metadata.num_rows
    if total_rows == 0:
        raise ValueError("文件没有任何记录")

    # 规范化负索引
    if index < 0:
        index = total_rows + index
    if index < 0 or index >= total_rows:
        raise IndexError(f"索引超出范围：{index}，总行数：{total_rows}")

    # 按行组定位到本地索引
    remaining = index
    target_row_group = 0
    local_index = 0
    for rg_idx in range(pf.metadata.num_row_groups):
        rg_rows = pf.metadata.row_group(rg_idx).num_rows
        if remaining < rg_rows:
            target_row_group = rg_idx
            local_index = remaining
            break
        remaining -= rg_rows

    # 仅读取目标行组
    table = pf.read_row_group(target_row_group)
    row = table.slice(local_index, 1).to_pylist()[0]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="查看 Parquet 文件中指定索引的一条完整记录（支持负数索引）",
    )
    parser.add_argument("path", help="Parquet 文件路径")
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=0,
        help="要查看的索引（从 0 开始，支持负数，默认 0）",
    )
    parser.add_argument(
        "--meta",
        action="store_true",
        help="同时打印文件元信息（行数、列数、行组数）",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON 缩进，默认 2",
    )

    args = parser.parse_args()

    try:
        pf = pq.ParquetFile(args.path)
        total_rows = get_total_rows(pf)
        num_row_groups = pf.metadata.num_row_groups if pf.metadata else 0
        num_columns = len(pf.schema.names) if hasattr(pf, "schema") else (
            pf.metadata.schema.num_columns if pf.metadata else 0
        )

        record = read_record_by_index(args.path, args.index)

        if args.meta:
            meta_info = {
                "file": args.path,
                "total_rows": int(total_rows),
                "num_columns": int(num_columns),
                "num_row_groups": int(num_row_groups),
                "requested_index": int(args.index),
            }
            print(json.dumps(meta_info, ensure_ascii=False, indent=args.indent))

        print(json.dumps(record, ensure_ascii=False, indent=args.indent, default=json_default_serializer))
        return 0
    except FileNotFoundError:
        print(f"文件不存在：{args.path}", file=sys.stderr)
        return 2
    except IndexError as e:
        print(str(e), file=sys.stderr)
        return 3
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 4
    except Exception as e:
        print(f"读取失败：{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


