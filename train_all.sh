#!/bin/bash

echo "开始执行所有train开头的Python文件..."
echo "========================================"

# 按文件名排序后依次执行
for file in train*.py; do
    if [ -f "$file" ]; then
        echo "执行: $file"
        echo "----------------------------------------"
        python "$file"
        
        # 检查上一个命令的退出状态
        if [ $? -eq 0 ]; then
            echo "✅ $file 执行成功"
        else
            echo "❌ $file 执行失败"
            # 是否继续执行？取消下面一行的注释来继续
            # continue
            # 或者停止执行
            
        fi
        
        echo "========================================"
    fi
done

echo "所有文件执行完成"