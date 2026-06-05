# NPU DSV4 Memory Pool 架构对比分析

对比 `g:\code\tmp\sglang`（tmp）与 `G:\code\sglang`（sglang）的 NPU DSV4 KV pool / allocator 实现。

---

## 核心差异：继承 vs 组合

| 维度 | tmp (`DSV4NPUTokenToKVPool`) | sglang (`SWAC4C128KVPool`) |
|------|------|------|
| **KV Pool 基类** | 继承 `DeepSeekV4TokenToKVPool`（CUDA 类） | 继承 `KVCache` 接口，不复用 CUDA 类 |
| **Allocator 基类** | 继承 `SWATokenToKVPoolAllocator` | 继承 `BaseTokenToKVPoolAllocator` |
| **ReqToTokenPool** | 专用子类 `DSV4NPUReqToTokenPool` | base `ReqToTokenPool` + `IS_DEEPSEEK_V4` env 分支扩展 |
| **State Pool 类型** | `NPUCompressStatePool`（继承 `CompressStatePool`） | `NPUSingleBufferTokenToKVPool`（通用 NPU pool 类） |
| **文件组织** | 4 个独立文件（memory_pool / state_pool / allocator / req_to_token） | 1 个文件（hybrid_swa_c4_c128_memory_pool.py）包含 KVPool + Allocator |

**根本原因**：tmp 的设计目标是与 SGLang 上游框架（CUDA path）保持架构兼容，通过继承复用逻辑。sglang 的设计是从零构建 NPU 专用池，不依赖 CUDA 基类。

---

## 1. KV Pool 对比

### tmp: `DSV4NPUTokenToKVPool`

```
继承 DeepSeekV4TokenToKVPool → 只 override _init_paged_compress_states 和 translate_kv_loc_to_compress_state_loc
```

- **共享**：所有 buffer 分配、属性设置（compression_ratios, swa_page_size, qk_nope_head_dim 等）由基类处理
- **仅改动**：state pool 从 `CompressStatePool` 换成 `NPUCompressStatePool`（paged 而非 ring）
- **translate 方法**：override 为 raise RuntimeError（ring-hash 在 NPU 无意义）
- **indexer KV pool**：有 `c4_indexer_kv_pool`（`DeepSeekV4IndexerPool`），NPU 下有独立的 `npu_index_k_buffer` / `npu_index_scale_buffer`（int8 + fp16 scale）
- **indexer state pool**：有 `indexer_compress_state_pools` 中的 c4 indexer state（`NPUCompressStatePool`，overlap=True），allocator 侧对应 `c4_index_state_attn_allocator`

### sglang: `SWAC4C128KVPool`

```
继承 KVCache → 手动创建 8 个 NPUSingleBufferTokenToKVPool 子 pool
```

- **不共享任何 CUDA 逻辑**：手动定义 size、size_swa、size_c4、size_c128 等参数
- **子 pool 列表**：dummy_kv_pool, swa_kv_pool, c4_kv_pool, c4_index_kv_pool, c4_state_pool, c4_index_state_pool, c128_kv_pool, c128_state_pool
- **提供**：get/set_swa_buffer, get/set_compress_buffer, get/set_compress_state_buffer 等方法
- **不支持**：translate_kv_loc_to_compress_state_loc（不存在此方法）
- **indexer KV pool**：有 `c4_index_kv_pool`（`NPUSingleBufferTokenToKVPool`，int8 dtype + dequant_scale_buffer）
- **indexer state pool**：有 `c4_index_state_pool`（`NPUSingleBufferTokenToKVPool`，float32, slot_dim=indexer_head_dim*2）

### 关键差异

| | tmp | sglang |
|------|------|------|
| indexer KV pool | `DeepSeekV4IndexerPool`（NPU 有 int8 k/scale buffer） | `NPUSingleBufferTokenToKVPool`（int8 dtype + dequant_scale） |
| indexer state pool | `NPUCompressStatePool`（overlap=True, 继承 CompressStatePool） | `NPUSingleBufferTokenToKVPool`（slot_dim=indexer_head_dim*2） |
| c128 state layout | `last_dim = 2*coff*head_dim = 2*head_dim`（KVAndScore wrapper） | `head_num=2, slot_dim=head_dim`（等效 `2*head_dim` per slot，布局语义不同） |
| block 0 sentinel | 有（kv=0, score=-inf） | 无 |
| state dtype | `CompressStatePool` 的 state_dtype | torch.float32（硬编码） |
| state 3D reshape | `state_cache_3d` property → `(num_blocks, page_size, last_dim)` | pool 本身无专用 property，call site 需要手动 view/reshape |

**两边都有 indexer KV pool 和 indexer state pool，只是类名和实现方式不同。**

---

## 2. Allocator 对比

### tmp: `DSV4NPUTokenToKVPoolAllocator`

```
继承 SWATokenToKVPoolAllocator → super() 处理 full + SWA，自己只管 c4/c128/state
```

- **alloc_extend**：调用 `super().alloc_extend()` 分配 full + SWA，然后 `_alloc_c_extend` + `_alloc_state_extend` 分配 c4/c128/state
- **alloc_decode**：调用 `super().alloc_decode()` 分配 full + SWA，然后同上
- **last_loc 来源**：allocator 自己通过 `get_last_loc()` 从 req_to_token 表查找（lazy）
- **返回值**：`out_full_loc`（与 CUDA 一致），bundle 通过 `get_last_dsv4_alloc()` 获取
- **free**：两种形式 — `free(free_index)` 和 `free(req=req, req_to_token_pool=pool)`
- **c4/c128 extend count**：allocator 内部 `_compute_c_extend_counts`

### sglang: `SWAC4C128TokenToKVPoolAllocator`

```
继承 BaseTokenToKVPoolAllocator → 手动管理 6 个子 allocator
```

- **alloc_extend**：直接调用 6 个子 allocator 的 `alloc_extend`
- **alloc_decode**：full/SWA/state 用 `alloc_decode`，c4/c128 用 `alloc_extend`（建模为 extend）
- **last_loc 来源**：调用者预先计算并传入 `LastLoc` 数据类（6 字段，explicit）
- **返回值**：`OutCacheLoc`（直接返回 bundle）
- **free**：`swa_c4_c128_free` 方法，传入所有 6 类 indices
- **c4/c128 extend count**：调用者传入 `ExtendNumTokens` 数据类

### 关键差异

| | tmp | sglang |
|------|------|------|
| full KV allocator | `full_attn_allocator`（从 super 继承） | `dummy_attn_allocator`（slot_dim=0，只占位） |
| last_loc 获取方式 | allocator 自己查找 req_to_token 表 | 调用者传入 LastLoc 数据类 |
| prefix/seq lens 传入方式 | c4/c128 KV 用 raw token 单位并在 allocator 内部转换；state lens 由 scheduler 预先计算后传入 | 每个维度分别传入（KvLen 数据类） |
| c4/c128 extend count | allocator 内部 `_compute_c_extend_counts` | 调用者传入 ExtendNumTokens |
| alloc 返回值 | `out_full_loc` + 侧信道 `get_last_dsv4_alloc()` | 直接返回 `OutCacheLoc` |
| free_group | 继承 super 的 free_group | 自己实现 `free_group_begin/end` |

**两边 state 都参与 allocator 分配和 req-to-token 写入，不是纯 ring-hash 推导。**

---

## 3. ReqToTokenPool 对比

### tmp: `DSV4NPUReqToTokenPool`

- 继承 `ReqToTokenPool`，新增 5 个辅助表：
  - `req_to_token_swa`（SWA slot ids）
  - `req_to_token_c4`（c4 compressed slot ids）
  - `req_to_token_c128`（c128 compressed slot ids）
  - `req_to_token_c4_state`（c4 state slot ids，per raw token）
  - `req_to_token_c128_state`（c128 state slot ids，per raw token）
- 有 `write_swa`, `write_c4`, `write_c128`, `write_c4_state`, `write_c128_state` 方法
- 有 `register_dsv4_allocator` 双向绑定 allocator
- `free(req)` 调用 allocator 的 `free(req=req, req_to_token_pool=self)`

### sglang: base `ReqToTokenPool` + env 分支

- 不用子类，而是在 `ReqToTokenPool.__init__` 中通过 `IS_DEEPSEEK_V4` 环境变量分支创建相同的辅助表：
  - `req_to_token_swa`, `req_to_token_c4`, `req_to_token_c128`
  - `req_to_token_c4_state`, `req_to_token_c128_state`
- 有 `write_c4_state`, `write_c128_state` 方法（在 base 类中）
- `mem_cache/common.py` 在 alloc 后同样写入这些表

**两边都有 per-req state table，只是扩展方式不同：tmp 用子类，sglang 用 env 分支。**

---

## 4. 类型系统对比

| | tmp | sglang |
|------|------|------|
| bundle 数据类 | `DSV4OutCacheLoc`（6 字段，2 个 Optional） | `OutCacheLoc`（6 字段，全 Optional，有工厂方法） |
| prefix/seq lens | 原始 tensor 参数（10+ kwarg） | `KvLen` 数据类（6 字段 + 工厂方法） |
| last_loc | 单个 `last_loc` tensor | `LastLoc` 数据类（6 字段 + 工厂方法） |
| extend count | 内部计算 | `ExtendNumTokens` 数据类（6 字段 + 工厂方法） |

sglang 用 4 个结构化数据类（KvLen, LastLoc, ExtendNumTokens, OutCacheLoc）封装所有维度，每个都有工厂方法和运算方法。tmp 用原始参数 + 数据类只封装最终 bundle。

---

## 5. 真正差异在哪里

**两边的功能覆盖基本一致**：都有 indexer KV pool、indexer state pool、per-req state table、c4/c128/state allocator。真正的差异不在"有没有某个功能"，而在：

### 5.1 Pool/Allocator 架构

- tmp 继承 CUDA 基类，只 override NPU 特有部分（state pool 类型 + translate 方法）
- sglang 完全独立实现，不继承 CUDA 类，所有功能手动重建

### 5.2 State Pool Buffer Layout

- tmp 用 `NPUCompressStatePool`（继承 `CompressStatePool`），提供 `KVAndScore` wrapper（kv/score 是连续 buffer 的两个半区）、`state_cache_3d` reshape、block 0 sentinel
- sglang 用 `NPUSingleBufferTokenToKVPool`，没有 kv/score 分割、没有 3D reshape、没有 block 0 sentinel。state pool buffer 形状等价但代码路径不同

### 5.3 数据类接口

- tmp：原始 tensor 参数传入 allocator；c4/c128 KV 的 prefix/seq/last_loc 在 allocator 内部转换和查表，c4/c128 state 的 prefix/seq/extend_num 由 scheduler 预先计算后通过 kwargs 传入
- sglang：结构化数据类（KvLen, LastLoc, ExtendNumTokens）显式传入所有维度的 prefix/seq/last_loc

### 5.4 Free/Evict 组织方式

- tmp：`free(req=req, req_to_token_pool=pool)` 一次调用释放所有子池槽位，继承 super 的 free_group
- sglang：`swa_c4_c128_free(dummy, swa, c4, c128, c4_state, c128_state)` 显式传入 6 类 indices，自己实现 free_group_begin/end

### 5.5 CUDA DSV4 基类兼容性

- tmp 保持兼容：NPU pool 是 CUDA pool 的子类，CUDA path 继续使用 `DeepSeekV4TokenToKVPool` 不受影响
- sglang 不兼容：NPU pool 是独立类，CUDA path 不受影响但 NPU path 无法跟随 CUDA 代码演进

---

## 6. sglang 的潜在问题

1. **没有 block 0 sentinel**：NPU kernel 的 cache_mode=1 用 block 0 作为 skip sentinel，未初始化的 state_block_table 默认指向 block 0。sglang 的 `NPUSingleBufferTokenToKVPool` 用 `torch.zeros` 初始化，所以 block 0 不是随机值；但它也不是显式的 attention-neutral sentinel（kv=0, score=-inf）。如果 kernel/上层逻辑把 block 0 当作可读 state 而不是纯 skip slot，全 0 score 可能在 softmax 中产生非零权重。

2. **state buffer 封装不贴近 compressor kernel**：fused compressor kernel (`torch.ops.custom.compressor` cache_mode=1) 需要 `state_cache` shape `(block_num, page_size, 2*coff*head_dim)`。sglang 的 `NPUSingleBufferTokenToKVPool` buffer shape 是 `(layer_num, size//page_size+1, page_size, head_num, slot_dim)`，call site 需要额外 view/reshape 才能传给 kernel。只要 call site 处理正确这不一定是 bug，但接口不如 tmp 的 `state_cache_3d` 直接。

3. **c128 state pool 布局语义不同**：sglang 用 `head_num=2, slot_dim=head_dim`（`(..., 2, head_dim)` 布局），tmp 用 `KVAndScore` wrapper（`(size, 2*head_dim)` 布局，kv 和 score 是连续半区）。数学等价但语义不同——tmp 的 kv/score 分割更直接匹配 compressor kernel 的读写模式，sglang 需要额外 reshape/transpose。

4. **IS_DEEPSEEK_V4 env 分支**：sglang 在 base `ReqToTokenPool` 中用全局 env 分支创建 DSV4 特有表，而不是子类化。这种做法让 base 类承担了 DSV4 特有逻辑，增加了 base 类的复杂度。tmp 的子类化更清晰——DSV4 特有表只在 `DSV4NPUReqToTokenPool` 中出现。

---

## 7. 为什么当前仓库的实现显得臃肿

tmp 的长期方向更合理，但当前实现确实显得比“继承式设计”本身更臃肿，主要原因不是功能多，而是兼容层和补丁层叠得比较厚：

1. **为保持 CUDA 接口兼容引入了侧信道**：`alloc_extend/alloc_decode` 仍返回 `out_full_loc`，DSV4 的 6 路 allocation bundle 通过 `get_last_dsv4_alloc()` 侧信道取出。这避免改动通用 allocator 调用栈，但让数据流不够直观。

2. **state lens 的职责分散**：c4/c128 KV 的 extend count 在 allocator 内部算，c4/c128 state 的 prefix/seq/extend_num 在 `ScheduleBatch._compute_dsv4_state_lens_*` 算，再经 `mem_cache/common.py` 透传给 allocator。结果是 scheduler、common、allocator 三处都知道 DSV4 state 的细节。

3. **ReqToTokenPool 和 allocator 双向绑定增加耦合**：`DSV4NPUReqToTokenPool.register_dsv4_allocator()` 同时设置 allocator back-ref，主要是为了 free 和 last_loc lookup。这能工作，但读代码时需要跨 pool/allocator/common/scheduler 跳转。

4. **NPU 兼容 shim 混在通用 DSV4 pool 中**：`DeepSeekV4IndexerPool` 同时保留 CUDA packed buffer 和 NPU 专用 `npu_index_k_buffer/npu_index_scale_buffer`，`DeepSeekV4SingleKVPool` 也在 NPU bf16 layout 和 CUDA uint8 packed layout 之间分支。单类承载两种 layout，局部复杂度上升。

5. **旧注释和新逻辑不一致**：`dsv4_allocator.py` / `dsv4_req_to_token_pool.py` 的模块 docstring 还描述 ring-hash state 逻辑，会让读者误以为 state 不走 paged allocator。

## 8. 如何精简当前仓库实现

可以保留“继承上游 CUDA DSV4 pool”的方向，同时把 NPU patch 收敛成更清晰的边界：

1. **把 DSV4 allocation bundle 显式化**：为通用 alloc helper 增加可选返回 bundle，或者统一返回 `(out_full_loc, aux)`。这样可以去掉 `get_last_dsv4_alloc()` 侧信道，`mem_cache/common.py` 不需要再从 allocator 读隐式状态。

2. **集中 state lens 计算**：把 `_compute_dsv4_state_lens_extend/decode` 产出的字段封装成一个小数据类，例如 `DSV4StateLens`，由 scheduler 只负责生成，common 只负责透传，allocator 只消费。现在 10+ 个 kwargs 可以缩成 1 个对象。

3. **给 NPU state pool 建一个统一 wrapper 接口**：抽象出 `get_state_cache(layer_id, from_indexer)`、`get_state_page_table(...)`、`set_state_buffer(...)` 这类方法，让 compressor/backend 不直接关心 `compress_state_pools` / `indexer_compress_state_pools` 的列表结构。

4. **把 NPU indexer buffer 从通用 `DeepSeekV4IndexerPool` 中拆成小策略类**：例如 `IndexerStorageCUDA` 和 `IndexerStorageNPU`，公共 pool 只暴露 `get_index_k`, `get_index_scale`, `set_index_k_scale`。这样可以减少 `if is_npu` 和 `npu_index_*` 字段散落。

5. **收敛 req_to_token 与 allocator 的双向依赖**：last_loc lookup 可以由 common 在写表处统一生成并传给 allocator，或者封装成 `DSV4ReqTables` helper。free 路径也可以改成由 tree/cache 层显式传 6 路 indices，而不是 `ReqToTokenPool.free(req)` 回调 allocator。

6. **先修文档和命名**：把 stale docstrings 改掉，把 `c4_indexer_kv_pool`、`indexer_compress_state_pools`、`c4_index_state_attn_allocator` 的关系写清楚。这是成本最低、收益最高的一步。

推荐的渐进路线：

1. 先修 stale docstrings 和命名注释。
2. 引入 `DSV4StateLens`，替换散落的 state kwargs。
3. 显式化 allocator bundle 返回，去掉 `get_last_dsv4_alloc()`。
4. 最后再考虑拆 `IndexerStorage` 策略类，避免一次性重构触碰太多热路径。

---

## 9. 建议

- **tmp 的继承式设计是更好的长期方向**：NPU path 自然跟随 CUDA codebase 演进，维护成本低。
- **tmp 应该修复 stale docstrings**：allocator 和 req_to_token_pool 的模块 docstring 仍然描述 CUDA ring-hash 逻辑，与实际 NPU paged 逻辑矛盾。
- **如果合并代码**：以 tmp 的继承式设计为基准。sglang 的功能（contiguous_buf_infos、free_group 等）可以逐步对齐到 tmp 的继承式接口，不需要替换为 sglang 的独立设计。
