下面方案有一份实现代码，请你检查代码是否存在问题？是否符合方案描述？

# MERGE：Mask-Edit-Refine **GE**neration

下表先给出所选组件一览，随后分别阐述**模型结构 → 训练流程 → 推理流程 → 关键超参 & 实践细节**。

| 子模块                  | 采用方案                                            | 关键理由                                   |
|----------------------|-------------------------------------------------|----------------------------------------|
| Gate                 | 置信度统计 + 轮次嵌入 → 2-层 MLP → Gumbel-Sigmoid ST      | ● 输入与“句子已成熟度”强相关 ● 可微、梯度稳定             |
| Mask-Picker (`M₁`)   | Soft Bottom-K 删除 + ST hard 分支                   | ● 与推理 Hard Bottom-K 对齐 ● `L_recon` 可回传 |
| Mask-Inserter (`M₂`) | **方案 A：Hard Multinomial ST + Mixed Sequence**   | ● 推理零落差 ● `L_recon` 直达 `M₂` 梯度         |
| 填充网络 (`E`)           | 12-L Transformer Encoder + MLM Head             | 处理软 / 混合序列                             |
| 训练信号                 | L_recon + sparsity & reward Gate 正则 + M₁/M₂ 熵正则 | ● 全部来自端到端梯度，最简监督                       |

---

## 1. 模型结构

### 1.1 Gate
```
features = concat(
mean_entropy(logits),# H̄
mean_margin(p1-p2),# 置信度差
low_conf_ratio(τ_H=1.5 nats),# 低置信比例
self_ppl,# ppl_self
round_emb(l) # 轮次嵌入
) # dim ≈ 5–10
g̃ = MLP2(features)# → ℝ
g̃ = sigmoid(g̃)
g_hard = ST(g̃, gumbel=True, τ)# τ 退火 2→0.2
```

### 1.2 Mask-Picker `M₁`

1. token-wise score
 `s_i = FFN(LLAMAdec(X_l))[i]`
2. soft 删除权重
 `p_del_i = σ(−s_i/τ_del)` (τ_del 同步退火)
3. **Soft token**
 `x̂_i = (1−p_del_i)·x_i + p_del_i·e[MASK]`
4. **Hard skeleton (ST)**
 ```
 idx_del= BottomK_ST(s_i, K_del)
 S_hard = X_l.delete(idx_del) # 长度 L−K
 ```

### 1.3 Mask-Inserter `M₂`(方案 A)

```
h_gap = GapEncoder(S_hard)# 得到 G=L−K+1 个 gap 表征
α_g = Linear(h_gap) # gap logits
z_g = softmax(α_g)# 分布
counts= Multinomial_ST(K_ins, z_g)# hard count (ST)
Ŝ_hard= insert_masks(S_hard, counts)# 真实插空
```

### 1.4 Mixed Sequence构造
```
Ŝ_mixed = fuse(Ŝ_hard, x̂_soft)
# 规则：若 token 为新插入 MASK → 用 e[MASK]
# 否则取 x̂_soft 对应位置的向量
```

### 1.5 填充网络 `E`
```
logits, hidden = E(Ŝ_mixed , t_l) # t_l = current_mask_ratio
```
Transformer + RoPE，相邻轮次参数共享；输出同时供 Gate 下一轮决策。

---

## 2. 训练流程

```python
for l in range(L):
# ------------ Gate ------------
g_hard, g_soft = Gate(hidden_prev)# 首轮 g=0

# ------------ Delete -----------
if g_hard==1:# Edit path
S_hard, x_soft = soft_bottomk_delete(X_l, K_del)
else:# Generate path
S_hard, x_soft = X_l, X_l# 无删除

# ------------ Insert -----------
K_ins = calc_k(|S_hard|) # 简单线性或上限20
Ŝ_hard= M2_insert(S_hard, K_ins)

# ------------ Fuse & Fill ------
Ŝ_mix = fuse(Ŝ_hard, x_soft)
logits, hidden = E(Ŝ_mix , t_l)

# ------------ Loss -------------
L_recon = weighted_CE(logits, Y*, t_l)# 仅 MASK 位
L_gate= λ_s·g_soft + λ_c·g_soft·relu(ΔH̄) # sparsity+reward
L_comp= λ_comp·KL(p_del ‖ Ber(r_target))
L_ins = λ_ins·Entropy(z_g)
loss= L_recon + L_gate + L_comp + L_ins
loss.backward(); optim.step()

X_l = teacher_force ? Y* : greedy_fill(Ŝ_hard, logits)
hidden_prev = hidden.detach()
```

* **teacher forcing**：前 10 epoch 用 `Y*`，后续 0.5 概率采样自生成。
* 退火：`τ_gate`、`τ_del` 线性 5 epoch → 0.2。
* 梯度裁剪 1.0；AdamW lr 3e-4。

---

## 3. 推理流程

```python
X = prompt_or_empty()
hidden_prev = zeros()
for l in range(L_max):
g = Gate_infer(hidden_prev) # τ_edit(l) = 0.4+0.1l
if g: # Edit
S = delete_bottomK(X, K_del)
else: # Generate
S = X
K_ins = calc_k(|S|)
Ŝ = insert_masks_generate(S, M2(S, K_ins))
logits, hidden = E(Ŝ , t_infer(mask_ratio(Ŝ)))
X_new, conf = greedy_fill(Ŝ, logits)

if conf>0.9 or lev(X_new,X)<2 or l==L_max-1:
break
X, hidden_prev = X_new, hidden
return detokenize(X_new)
```

---

## 4. 关键超参

```
L_max= 6
K_del= round(0.15 · L)∈ [1,30]
calc_k(|S|)= clamp(round(0.25·(|Y*|−|S|)), 1, 20)
λ_s= 0.1 # Gate sparsity
λ_c= 0.2 # Gate reward
λ_comp = 0.05# 删除 KL
λ_ins= 0.01# 插空熵正则
```

---

## 5. 实践 Tips

1. **冻结顺序**
 ① 先训练 `E`(full-mask MLM) 2-3 epoch → ② 打开 `M₁/M₂` → ③ 最后解冻 Gate。
2. **显存节省**：软删除可在 embedding 层完成，无需新张量。
3. **稳定 Multinomial_ST**：若 `K_ins>8`，把一次插空拆成多轮，每轮 ≤8，梯度噪声更小。
4. **对齐混合**：`fuse` 函数应写成纯 `torch` 索引，确保 ST 的 `.grad_fn` 保留。

---

## 6. 方案优势回顾

* **与推理同形**：`M₁` Hard Bottom-K、`M₂` Hard Multinomial ST 全程与部署一致。
* **端到端可微**：Soft-Delete & ST 让 `L_recon` 的梯度到达所有子模块。
* **训练信号简洁**：仅 `L_recon` + 小量正则，无需额外对齐标签。
* **质量与效率兼顾**：Gate 自适应决定是否微调句子，避免无谓迭代；`E` 使用 Encoder 权重对高 mask 比例也稳健。
