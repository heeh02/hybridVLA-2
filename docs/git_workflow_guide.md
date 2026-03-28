# Git 工作流新手指南

> 本文档面向 Git 初学者，解释本项目使用的 Git 工作流、分支策略、提交规范等概念。
> 所有操作都以命令行为主，配合 GitHub 使用。

---

## 目录

1. [核心概念速览](#1-核心概念速览)
2. [分支模型详解](#2-分支模型详解)
3. [Commit 提交规范](#3-commit-提交规范)
4. [Tag 与版本号](#4-tag-与版本号)
5. [CHANGELOG 变更日志](#5-changelog-变更日志)
6. [GitHub Actions CI 持续集成](#6-github-actions-ci-持续集成)
7. [日常开发工作流（完整示例）](#7-日常开发工作流完整示例)
8. [常用 Git 命令速查](#8-常用-git-命令速查)
9. [出问题了怎么办](#9-出问题了怎么办)

---

## 1. 核心概念速览

### Git 是什么？

Git 是一个**版本控制系统**。你可以把它想象成一个超级"撤销"按钮——不仅能撤销，
还能看到项目的任何一个历史状态，甚至可以同时维护多个不同的版本。

### 几个关键概念

| 概念 | 比喻 | 说明 |
|------|------|------|
| **Repository (仓库)** | 项目文件夹 | 包含所有代码和完整历史记录 |
| **Commit (提交)** | 存档点 | 代码在某一时刻的快照，有唯一 ID |
| **Branch (分支)** | 平行宇宙 | 从某个点分出去独立开发，互不影响 |
| **Merge (合并)** | 宇宙合并 | 把一个分支的改动合并到另一个分支 |
| **Tag (标签)** | 里程碑标记 | 给某个 commit 起一个永久的名字（如 v1.0.0） |
| **Remote (远程)** | 云端备份 | GitHub 上的仓库副本 |
| **Push (推送)** | 上传 | 把本地的 commit 推送到 GitHub |
| **Pull (拉取)** | 下载 | 把 GitHub 上的新 commit 拉到本地 |

---

## 2. 分支模型详解

### 我们的分支结构

```
main  ←── 稳定版本，只接受从 dev 的合并，打 tag 发布
 │
 └── dev  ←── 日常开发主线
      │
      ├── feature/xxx   ←── 新功能（完成后合并回 dev，然后删除）
      ├── fix/xxx       ←── 修复 bug
      └── experiment/xxx ←── 实验性尝试（可能不会合并）
```

### 每个分支的作用

**`main` 分支** — 生产线 / 发布线
- 永远保持**可用、稳定**的状态
- **绝不直接在 main 上写代码**
- 只通过合并 `dev` 分支来更新
- 每次合并后打一个版本 tag（如 v1.1.0）

**`dev` 分支** — 开发主线
- 你的日常工作都在这里或者从这里分出去的分支上进行
- 代码可能不完美，但应该**能跑**
- 积累了足够的功能后，合并到 `main` 发布

**`feature/xxx` 分支** — 功能分支
- 每开发一个新功能，从 `dev` 拉出一个新分支
- 命名格式：`feature/描述`，如 `feature/multi-camera`
- 开发完成后，合并回 `dev`，然后删除这个分支

**`fix/xxx` 分支** — 修复分支
- 修 bug 时从 `dev` 拉出
- 如 `fix/mamba-memory-leak`
- 修完合并回 `dev`

**`experiment/xxx` 分支** — 实验分支
- ML 研究项目特有的——用来尝试新想法
- 如 `experiment/cosine-noise-schedule`
- 结果好就合并，不好就留着做记录或直接删除

### 操作示例

```bash
# ──────────── 开始一个新功能 ────────────

# 1. 确保 dev 是最新的
git checkout dev
git pull origin dev

# 2. 从 dev 创建功能分支
git checkout -b feature/multi-camera

# 3. 在功能分支上写代码、提交
#    ... 写代码 ...
git add vla_hybrid_v2/models/multi_camera.py
git commit -m "feat: add multi-camera fusion module"

#    ... 继续写 ...
git add tests/test_multi_camera.py
git commit -m "test: add multi-camera unit tests"

# 4. 功能完成，合并回 dev
git checkout dev
git merge feature/multi-camera

# 5. 删除功能分支（已合并，不再需要）
git branch -d feature/multi-camera

# 6. 推送 dev 到 GitHub
git push origin dev


# ──────────── 发布新版本 ────────────

# 1. dev 上功能稳定了，合并到 main
git checkout main
git merge dev

# 2. 打版本 tag
git tag -a v1.1.0 -m "v1.1.0: add multi-camera support"

# 3. 推送 main 和 tag 到 GitHub
git push origin main
git push origin v1.1.0

# 4. 更新 CHANGELOG.md（见第 5 节）
```

### 为什么不直接在 main 上开发？

想象你正在 `main` 上开发一个新功能，写到一半发现线上出了个紧急 bug。
如果你直接在 `main` 上开发，你的半成品代码和紧急修复混在一起，一团糟。

用分支模型：
- 你在 `feature/xxx` 上开发新功能
- 紧急 bug 出现 → 切回 `dev`，拉出 `fix/urgent-bug`，修复，合并
- 然后切回 `feature/xxx` 继续开发
- 互不影响

---

## 3. Commit 提交规范

### 什么是好的 Commit？

一个好的 commit 应该是：
- **原子性的**：只做一件事
- **可描述的**：看 message 就知道做了什么
- **可回滚的**：出问题时能单独撤回这个改动

### Conventional Commits 格式

我们使用 **Conventional Commits** 规范，格式为：

```
<类型>: <简短描述>

<可选的详细说明>
```

### 类型对照表

| 类型 | 含义 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat: add multi-step supervision` |
| `fix` | 修复 bug | `fix: resolve FSDP gradient sync issue` |
| `refactor` | 重构（不改功能） | `refactor: move world model to experimental/` |
| `test` | 测试相关 | `test: add EMA/FSDP gap tests` |
| `docs` | 文档 | `docs: update README with training instructions` |
| `perf` | 性能优化 | `perf: optimize Mamba selective scan memory` |
| `ci` | CI/CD 相关 | `ci: add GitHub Actions workflow` |
| `chore` | 杂项（依赖、配置） | `chore: upgrade torch to 2.2.0` |
| `data` | 数据处理相关 | `data: add LIBERO HDF5 dataset adapter` |
| `experiment` | 实验性改动 | `experiment: test cosine noise schedule` |

### 好 vs 坏 的 Commit Message

```bash
# ❌ 坏的——看不出做了什么
git commit -m "update"
git commit -m "fix bug"
git commit -m "changes"
git commit -m "asdfg"

# ✅ 好的——一目了然
git commit -m "feat: add multi-step supervision to flow matching loss"
git commit -m "fix: resolve FSDP gradient sync issue on multi-GPU"
git commit -m "refactor: simplify train_stage_a by removing redundant boilerplate"
```

### 什么时候应该 Commit？

- 完成了一个**完整的小改动**（比如写完一个函数、修好一个 bug）
- 代码**能运行**（至少不报 import 错误）
- 你要去做**另一件不相关的事**之前

不要：
- 攒一天的改动一次性 commit
- 把 bug 修复和新功能混在同一个 commit 里

---

## 4. Tag 与版本号

### 语义化版本号 (Semantic Versioning)

版本号格式：**MAJOR.MINOR.PATCH**，即 `主版本.次版本.补丁版`

```
v1.2.3
│ │ └── PATCH (补丁): 修 bug，向后兼容
│ └──── MINOR (次版本): 加功能，向后兼容
└────── MAJOR (主版本): 有破坏性变更，不向后兼容
```

### 什么时候升什么版本？

| 变更类型 | 版本变化 | 示例 |
|---------|---------|------|
| 修了一个 bug | PATCH +1 | 1.0.0 → 1.0.1 |
| 加了新功能（不影响旧功能） | MINOR +1 | 1.0.0 → 1.1.0 |
| 重大重构 / 接口变更 | MAJOR +1 | 1.0.0 → 2.0.0 |

> MINOR 升级时 PATCH 归零（1.0.3 → 1.1.0）
> MAJOR 升级时 MINOR 和 PATCH 都归零（1.3.2 → 2.0.0）

### Tag 操作

```bash
# 查看所有 tag
git tag -l

# 创建带注释的 tag（推荐）
git tag -a v1.1.0 -m "v1.1.0: add multi-camera support"

# 推送 tag 到 GitHub
git push origin v1.1.0

# 推送所有 tag
git push origin --tags

# 查看某个 tag 的详细信息
git show v1.0.0

# 切换到某个 tag 对应的代码状态（只读）
git checkout v1.0.0
```

### Tag 和 Branch 的区别

- **Branch** 是活动的，会随着新 commit 向前移动
- **Tag** 是固定的，永远指向创建时的那个 commit
- Tag 就像书签，标记了"这个版本的代码长这样"

---

## 5. CHANGELOG 变更日志

### 它是什么？

CHANGELOG.md 是一个**人类可读**的文件，记录了每个版本的变更内容。
不同于 git log（面向开发者的原始记录），CHANGELOG 是给**用户和自己**看的摘要。

### 格式（Keep a Changelog）

```markdown
## [1.1.0] - 2026-04-15

### Added（新增）
- 多相机融合模块

### Changed（变更）
- 优化 Mamba 内存使用

### Fixed（修复）
- 修复 FSDP 梯度同步问题

### Removed（移除）
- 删除废弃的 legacy 训练脚本
```

### 分类说明

| 分类 | 用于 |
|------|------|
| **Added** | 全新的功能 |
| **Changed** | 对已有功能的改动 |
| **Deprecated** | 即将移除的功能（给用户警告） |
| **Removed** | 已经移除的功能 |
| **Fixed** | bug 修复 |
| **Security** | 安全相关的修复 |

### 什么时候更新？

每次发布新版本（往 `main` 合并并打 tag）时，更新 CHANGELOG。
养成习惯：**打 tag 之前先更新 CHANGELOG**。

---

## 6. GitHub Actions CI 持续集成

### 它是什么？

CI (Continuous Integration) = 每次推代码到 GitHub，自动帮你运行测试和检查。
不用手动跑 `pytest`，GitHub 会在云端帮你跑，结果直接显示在 PR 页面上。

### 我们的 CI 配置

文件位置：`.github/workflows/ci.yml`

```yaml
name: CI

on:                          # 什么时候触发？
  push:
    branches: [main, dev]    #   推送到 main 或 dev 时
  pull_request:
    branches: [dev]          #   向 dev 提 PR 时

jobs:
  lint:                      # 任务 1：代码风格检查
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4       # 拉取代码
      - uses: actions/setup-python@v5   # 安装 Python
        with:
          python-version: "3.10"
      - run: pip install ruff           # 安装 linter
      - run: ruff check .              # 检查代码错误
      - run: ruff format --check .     # 检查代码格式

  test:                      # 任务 2：运行测试
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
          cache: pip                    # 缓存依赖，加速
      - run: pip install -r requirements.txt pytest
      - run: pytest                    # 运行所有测试
```

### 它会做什么？

1. 你 `git push` 到 `dev` 或 `main`
2. GitHub 自动启动一台云服务器
3. 在上面安装 Python 和你的依赖
4. 运行 `ruff check`（检查代码有没有语法错误、import 顺序等）
5. 运行 `ruff format --check`（检查代码格式是否规范）
6. 运行 `pytest`（跑所有测试）
7. 全部通过 → 绿色 ✓；任何一步失败 → 红色 ✗

### 在哪里看结果？

- GitHub 仓库页面 → "Actions" 标签页
- 每个 PR 页面的底部会显示 CI 状态
- commit 旁边会出现绿色 ✓ 或红色 ✗

---

## 7. 日常开发工作流（完整示例）

下面是一个从头到尾的完整开发流程示例：

```bash
# ======================================
# 场景：添加"多相机支持"功能
# ======================================

# 一、准备工作
git checkout dev                  # 切换到 dev 分支
git pull origin dev               # 拉取最新代码
git checkout -b feature/multi-cam # 创建功能分支

# 二、开发（在功能分支上）
# 写代码 → 测试 → 提交，循环

# 写了相机融合模块
git add vla_hybrid_v2/models/multi_camera.py
git commit -m "feat: add multi-camera fusion module"

# 写了对应的配置
git add vla_hybrid_v2/config.py
git commit -m "feat: add multi-camera config options"

# 写了测试
git add tests/test_multi_camera.py
git commit -m "test: add multi-camera unit tests"

# 跑一下测试确认没问题
pytest tests/test_multi_camera.py

# 三、合并回 dev
git checkout dev                     # 切回 dev
git merge feature/multi-cam          # 合并功能分支
git branch -d feature/multi-cam      # 删除功能分支
git push origin dev                  # 推到 GitHub
# → GitHub Actions 自动运行 CI

# 四、发布（积累了几个功能后）
git checkout main                    # 切到 main
git merge dev                        # 合并 dev

# 更新 CHANGELOG.md（手动编辑，添加新版本的条目）
git add CHANGELOG.md
git commit -m "docs: update CHANGELOG for v1.1.0"

# 打 tag
git tag -a v1.1.0 -m "v1.1.0: multi-camera support, bug fixes"

# 推送
git push origin main
git push origin v1.1.0

# 五、同步 dev
git checkout dev
git merge main                       # 让 dev 也包含 CHANGELOG 等变更
git push origin dev
```

### 流程图

```
              feature/multi-cam
             ╱                 ╲
dev ────●───●                   ●───●──── dev
                                     ╲
main ─────────────────────────────────●── main (tag: v1.1.0)
```

---

## 8. 常用 Git 命令速查

### 基础操作

```bash
git status                    # 查看当前状态（有哪些改动）
git log --oneline -10         # 查看最近 10 条 commit
git diff                      # 查看未暂存的改动
git diff --staged             # 查看已暂存（add 过的）改动
```

### 分支操作

```bash
git branch                    # 查看所有本地分支
git branch -a                 # 查看所有分支（含远程）
git checkout <分支名>          # 切换分支
git checkout -b <新分支名>     # 创建并切换到新分支
git branch -d <分支名>        # 删除已合并的分支
git merge <分支名>            # 把指定分支合并到当前分支
```

### 提交操作

```bash
git add <文件>                # 暂存文件（准备提交）
git add -A                    # 暂存所有改动（谨慎使用！）
git commit -m "message"       # 提交
git push origin <分支名>      # 推送到 GitHub
git pull origin <分支名>      # 从 GitHub 拉取
```

### Tag 操作

```bash
git tag -l                    # 列出所有 tag
git tag -a v1.0.0 -m "msg"   # 创建 tag
git push origin v1.0.0        # 推送 tag
git push origin --tags        # 推送所有 tag
```

### 查看历史

```bash
git log --oneline --graph     # 图形化显示分支历史
git show <commit-id>          # 查看某次 commit 的详情
git blame <文件>              # 查看文件每一行是谁写的
```

---

## 9. 出问题了怎么办

### "我还没 commit，但改坏了一个文件，想恢复"

```bash
# 恢复单个文件到最后一次 commit 的状态
git checkout -- <文件路径>
```

### "我 commit 了但写错了 message"

```bash
# 修改最近一次 commit 的 message（还没 push 的情况下）
git commit --amend -m "新的 message"
```

### "我 commit 了但内容有误，想撤回"

```bash
# 撤回最近一次 commit，但保留文件改动（推荐）
git reset --soft HEAD~1

# 撤回最近一次 commit，文件改动也丢弃（危险！）
git reset --hard HEAD~1
```

### "合并时出现冲突了"

```
<<<<<<< HEAD
你的代码
=======
对方的代码
>>>>>>> feature/xxx
```

Git 会在冲突文件中标记出冲突位置。你需要：
1. 打开文件，手动选择保留哪部分（或两者结合）
2. 删除 `<<<<<<<`、`=======`、`>>>>>>>` 标记
3. `git add <冲突文件>`
4. `git commit` 完成合并

### "我在错误的分支上开发了"

```bash
# 还没 commit：把改动带到正确的分支
git stash                     # 暂存当前改动
git checkout correct-branch   # 切换到正确分支
git stash pop                 # 恢复改动

# 已经 commit 了：用 cherry-pick 搬运
git checkout correct-branch
git cherry-pick <commit-id>   # 把那个 commit 搬过来
```

---

## 附：本项目当前结构

```
hybridVLA_2/
├── .github/workflows/ci.yml   ← CI 配置（自动测试）
├── CHANGELOG.md                ← 版本变更记录
├── pyproject.toml              ← 项目配置（版本号、测试配置、linter 配置）
├── requirements.txt            ← 依赖列表
├── scripts/                    ← 训练脚本
├── tests/                      ← 测试文件
├── docs/                       ← 文档
└── vla_hybrid_v2/              ← 主代码
    ├── models/                 ← 模型定义
    ├── losses/                 ← 损失函数
    ├── ops/                    ← 底层算子
    ├── utils/                  ← 工具函数
    ├── infer/                  ← 推理代码
    └── experimental/           ← 实验性代码（如 world model）
```

### 分支状态

| 分支 | 作用 | 远程 |
|------|------|------|
| `main` | 稳定发布线 | `origin/main` |
| `dev` | 日常开发线 | 需要 `git push -u origin dev` |

### 版本 Tag

| Tag | 说明 |
|-----|------|
| `v0.10.3` | HDF5 读取、多步监督 |
| `v0.10.5` | 配置修复、数据改进 |
| `v0.10.7` | LIBERO 集成、测试套件 |
| `v0.10.9` | 推理策略、FSDP 修复 |
| `v1.0.0` | 架构重构、代码审查修复（当前） |
