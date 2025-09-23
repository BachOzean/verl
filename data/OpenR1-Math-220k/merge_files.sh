#!/bin/bash

echo "开始合并分片文件..."

# 检查分片文件是否存在
if ls openr1-math-220k.tar.gz.part-* 1> /dev/null 2>&1; then
    echo "找到分片文件，开始合并..."
    
    # 合并所有分片文件
    cat openr1-math-220k.tar.gz.part-* > openr1-math-220k_restored.tar.gz
    
    # 验证文件完整性
    if [ -f "openr1-math-220k_restored.tar.gz" ]; then
        echo "合并完成！"
        echo "原始文件大小: $(du -h openr1-math-220k.tar.gz | cut -f1)"
        echo "合并后大小: $(du -h openr1-math-220k_restored.tar.gz | cut -f1)"
        
        # 可选：验证文件完整性
        echo "验证文件完整性..."
        if [ -f "openr1-math-220k.tar.gz" ]; then
            original_md5=$(md5sum openr1-math-220k.tar.gz | cut -d' ' -f1)
            restored_md5=$(md5sum openr1-math-220k_restored.tar.gz | cut -d' ' -f1)
            
            if [ "$original_md5" = "$restored_md5" ]; then
                echo "✅ 文件验证成功，内容一致！"
            else
                echo "❌ 文件验证失败，内容不一致！"
            fi
        else
            echo "⚠️  原始文件不存在，跳过完整性验证"
        fi
    else
        echo "❌ 合并失败！"
        exit 1
    fi
else
    echo "❌ 未找到分片文件！"
    exit 1
fi

echo "使用方法:"
echo "1. 下载所有分片文件到同一目录"
echo "2. 运行: bash merge_files.sh"
