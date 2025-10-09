#!/bin/bash
# 批量创建书籍章节框架

cd book_chapters

# 第2-17章（历史和现状部分）- 创建框架
for i in {02..17}; do
    cat > ${i}_chapter.md << CHAPTER
# 第${i#0}章节框架

[本章内容基于原始大纲，待详细编写]

## 核心内容

- 历史背景
- 技术发展
- 案例分析  
- 学术点睛
- 与EIT-P的联系

## 本章小结

## 思考题

## 延伸阅读

---

*本章完*
CHAPTER
    echo "✅ 创建第${i#0}章框架"
done

echo ""
echo "✅ 历史章节框架创建完成"
echo "📝 现在专注于核心EIT-P章节（18-20章）的详细编写"
