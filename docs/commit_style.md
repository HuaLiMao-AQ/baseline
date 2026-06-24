# 提交建议

提交信息保持短、自然、可读。优先让人一眼看懂这次改了什么，不为了格式牺牲可读性。

## 推荐写法

- 初始化项目可以直接用 `Initial commit`。
- 小修复可以写 `Fix artifact validation JSON output`。
- 结构整理可以写 `整理指标辅助函数`。
- 数据或结果整理可以写 `Add baseline analysis tables`。
- 中文提交也可以，例如 `整理 baseline 结果说明`。

## 可选 Conventional Commit

如果某次改动很适合 Conventional Commit，可以使用：

```text
feat(dataset): add jsonl sample loader
fix(parser): handle fenced json output
docs(readme): 更新运行说明
```

但这不是硬要求。
