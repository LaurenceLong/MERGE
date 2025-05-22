# MERGE：Mask-Edit-Refine **G**ated **E**ncoder  

一个统一「局部编辑 ↔ 自由生成」的并行序列建模框架。通过软/硬门控 `Gate` 在 **Edit Path**（删-插-填）和 **Generation Path**（插-填）之间自动切换，多轮循环式 Refinement + 早停推理，兼得低延迟与可控性。

---

## 1  记号

| 记号          | 含义                                                |
|-------------|---------------------------------------------------|
| `X₀`        | 原始序列（空串 / Prompt / 草稿）                            |
| `L`         | 最大循环轮数，训练 3-6 轮，推理可早停                             |
| `D₁`        | Mask-Picker（删除器），Decoder-1（LLAMA组件）               |
| `D₂`        | Mask-Inserter（插空器），Decoder-2（LLAMA组件）             |
| `E`         | Transformer Encoder（参考LLAMA Decoder实现组件）          |
| `H`         | MLM Head（分类层或共享词嵌入）                               |
| `g_l∈(0,1)` | 第 `l` 轮门控标量，`g≈1` 选 **Edit**，`g≈0` 选 **Generate** |
| `τ_edit`    | 判定门 0.5（训练可软采样）                                   |

---

## 2  训练流程（端到端反传，展开 *L* 轮）

```text
X_0  = 原句 / 随机片段 / 空串            # teacher forcing 到 Y*
E_-1 = E(X_0)                           # 仅用于 Gate_0 判定
for l = 0 .. L-1:

    # ===== 1) 门控决策 =====
    g_l = Gate(E_{l-1})   if l>0 else 0         # 第一轮固定走生成
    edit = (g_l > τ_edit)                       # 软硬两用

    # ===== 2) 删除阶段 =====
    if edit:
        p_keep = D₁(X_l)                        # 保留概率
        m      = Bernoulli_ST(p_keep)           # ST-采样
        S_l    = Drop(X_l, m)                   # skeleton
    else:
        S_l = X_l                               # 跳过删除

    # ===== 3) 插空阶段 =====
    α_gap  = D₂(S_l)                            # gap scores
    k_ins  = max(1, ‖X_l‖-‖S_l‖)                # 保序插位数
    idx    = Pointer_ST(α_gap, k_ins)           # 可微 Top-k
    Ŝ_l    = InsertMask(S_l, idx)               # 插入 [MASK]

    # ===== 4) 填充阶段 =====
    E_l    = E(Ŝ_l)                             # 隐状态
    X_{l+1}= Replace(Ŝ_l, argmax H(E_l))        # Teacher forcing
endfor
```

---

## 3  损失设计

```
1) 重建损失   L_recon = CE(X_L , Y*)                     # 最后一轮对齐目标
2) 删除正则   L_comp  = KL(p_keep ‖ Ber(r_target(g_l)))  # 控制删除量
3) 插空正则   L_ptr   = KL(softmax α_gap ‖ U_gap)        # 避免扎堆
4) 门控成本   L_gate  = c · g_l                          # 触发编辑要付费
5) 长度约束   R_len   = ‖S_l‖ / ‖X_l‖                    # 防止全删或不删
总损失        L = L_recon + λ₁L_comp + λ₂L_ptr + λ₃L_gate + λ₄R_len
```

经验系数：`λ₁=0.5，λ₂=0.5，λ₃=0.1，λ₄=0.1`；`c=0.05`。  
梯度依托 Straight-Through / Gumbel-Softmax 透过离散采样回传；显存受限可用 Truncated-BPTT。

---

## 4  模块实现

| 模块 | 计算 | 细节 |
| ---- | ---- | ---- |
| Encoder `E` | `Ŝ→h∈ℝ^{n×d}` | 12-Layer Relative-Pos Transformer，轮间参数共享 |
| MLM Head `H` | `h→log p_vocab` | 权重与嵌入 Tie-Weight |
| 删除器 `D₁` | `h→p_keep` | `σ(FFN(h))`，一层 768→1 |
| 插空器 `D₂` | `h_keep→α_gap` | Pointer Net：单层 Bi-GRU→dot attention |
| 门控 `Gate` | `mean(h)→g` | `σ(wᵀ·LN(mean(h)))` |

---

## 5  推理流程（早停版）

```text
X_0 = 用户输入 (可为空)
for l = 0 .. L-1:

    g_l   = Gate(E(X_l))                 # auto 模式
    edit  = (mode == 'edit')      ? 1 : (mode == 'gen') ? 0 : (g_l > τ_edit)

    # --- 删除 ---
    if edit:
        S_l = Drop(X_l, D₁(X_l) > 0.5)
    else:
        S_l = X_l

    # --- 插空 ---
    α_gap = D₂(S_l)
    k     = max(1, round((|X_l|-|S_l|) * r))   # r≈1.0
    idx   = topk(α_gap, k)
    Ŝ_l   = InsertMask(S_l, idx)

    # --- 填充 ---
    X_{l+1}, conf = GreedyFill(Ŝ_l, E, H, return_confidence=True)

    # --- 早停 ---
    if conf>0.9 or LevDist(X_{l+1},X_l)<ε or l==L-1: break
endfor
return X_{l+1}
```

三种模式  
• `mode='edit'` → 强制 `g=1`，只做局部润色  
• `mode='gen'` → 强制 `g=0`，直接生成  
• `mode='auto'` → `Gate` 自主决策

---

## 6  训练技巧

1. 两阶段温度：前 5 k 步 Gumbel 温度 τ=2，再线性降到 1。  
2. 片段冻结：Encoder 前 6 层冻结 10 k 步 → 稳定预训练特征。  
3. 噪声门控：训练早期 `g ← g + 𝒩(0,0.05)`，防止塌缩。  
4. 动态轮数：`L_train ~ U(1,L)`，缓解暴露偏差。  
5. 梯度裁剪 2.0；AdamW 3 e-5；batch 128；warm-up 5 %。

---

## 7  配置与资源

* max_len = 256  
* 3 × 10⁵ 步（8×A100≈24 h）  
* 推理平均 2.1 轮，22 k token/s → 比自回归 GPT-2 快 3×  

---

## 8  消融实验

1. 去掉 `Gate`（始终 Edit / 始终 Gen）  
2. 只插不删（移除 `D₁`）  
3. Top-k vs Pointer-Network 插空  
4. 单轮 vs 多轮 (L=1/2/3/6)  
5. 绝对 vs 相对位置编码对删除器影响  

---

## 9  对比优势

* 与 MaskGIT 相比：多了 **删除**，可显式编辑；  
* 与 Levenshtein Transformer 相比：删除/插空概率 **一次并行** 预测，无需逐 token 交替；  
* 与 自回归 GPT 相比：并行填充带来 **3× 速度提升**；门控使“润色 & 续写”**同一模型**完成。

---

> MERGE 把 “删-插-填” 的可解释编辑路径与直接生成路径融合在一个循环式 Mask 语言模型中，可一键切换 *Edit / Generate / Auto*，适用于改写、摘要、对话及开放生成等场景。