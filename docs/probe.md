OPD-Aha：Aha Finding 是如何被 Probe 出来的？

论文：OPD-Aha: Learning to Reconsider Visual Evidence through Reconstructed Privileged Supervision
目标：只整理“作者是怎么通过 probe 发现问题、验证 finding、再验证机制”的过程。
重点：输入数据是什么、模型怎么放、如何采集状态、保留 prefix 是什么意思、比较什么量、控制变量是什么、每个 probe 想排除什么解释。

0. 先说结论：作者的 Aha 不是直接从训练结果里“撞出来”的

这篇工作的核心 insight 并不是：

“我们训练了一个新 loss，发现分数更高。”

作者先对 已经失败的 Vision-OPD trajectory 做离线 replay / probe，先发现两个连续现象：

Prefix dominance：Student 已经写出来的错误 reasoning 越长，privileged Teacher 越容易被这个错误语言上下文带跑，即使 Teacher 拿着更好的视觉证据。

Hidden evidence response：但 Teacher 被带跑以后，视觉证据对 Teacher 的 token preference 并没有完全消失；这个影响只是不能再通过 Teacher - Student 的 prediction gap 看出来。把同一个 Teacher 的 real privileged image 和 visual-null image 做 paired comparison，仍然可以读出一个指向正确答案的视觉 correction。

因此真正的 Aha 是：

Teacher 不是“不知道正确方向了”，而是正确的视觉方向被已有错误 prefix 压在了 observable prediction 下面。

所以后面的 OPD-Aha 方法才顺理成章：

[
\text{不要只 distill }p_T^+,
\quad
\text{而要把 }(\log p_T^+ - \log p_T^0)
\text{ 里隐藏的视觉 preference 加回来。}
]

1. Probe 前的基本对象：一条 Vision-OPD trajectory 里有什么？

对一条视觉推理样本，可以抽象成：

原始完整图像：(I)

问题：(x)

训练时 Teacher 可见的 privileged localized crop：(I^+)

Student 自己生成的 response：

[
y=(y_1,y_2,\ldots,y_T)
]

正确答案序列：(z^\star)

Student 最终实际给出的错误答案序列：(z^{err})

Student 在生成第 (t) 个 token 时，已经有语言 prefix：

[
y_{<t}=(y_1,\ldots,y_{t-1})
]

把 question 和 prefix 合起来：

[
c_t=(x,y_{<t})
]

这是整篇论文最重要的状态变量。

1.1 Student 与 Teacher 的关键不对称

标准 privileged multimodal OPD 中：

Student

Student 看原始完整图：

[
p_t^S(v)=p_\theta(v\mid I,c_t)
]

Teacher

Teacher 看更有信息的 localized privileged crop：

[
p_t^+(v)=p_\phi(v\mid I^+,c_t)
]

Teacher 在视觉上有优势，但两者共享同一个：

[
c_t=(x,y_{<t})
]

而这个 (y_{<t}) 是 Student 自己写出来的。

这就是后面所有 probe 的核心：
只改变 prefix 长度，或者只改变视觉证据，其他条件尽量保持不变。

2. 模型设置：Probe 到底是在什么模型上做的？

论文给出的整体实验配置是：

项目

设置

Student backbone

Qwen3.5-4B / Qwen3.5-9B

Teacher

initial Student 的 frozen copy

Training data

Vision-OPD-6K，6,241 examples

Student visual input

original full image

Teacher evidence-present input

localized evidence crop

Teacher evidence-absent input

和 crop 尺寸一致的 mean-RGB image

Distillation

token-level Jensen–Shannon divergence

Distribution support

Student top-100 tokens + one tail bucket

Rollouts / prompt

8

Max response length

1,024 tokens

但是要特别区分：

论文明确给出的

上述是 OPD-Aha 整体训练/实验的共享配置。

当前 draft 没有明确给出的

Figure 1 的核心 diagnostic probe 没有在正文中明确写出：

Figure 1 使用的是 4B 还是 9B；

使用哪个具体 checkpoint；

一共收集了多少条 failed trajectories；

failed trajectories 来自哪一个 benchmark split、各类各多少条；

retention 时 token 数如何 rounding；

“common answer prompt”的精确文本；

Figure 1 上曲线的具体 smoothing / binning 实现。

因此复现时不能把这些未写出的细节当成论文已声明事实。

更稳妥的理解是：

作者拿 failed Vision-OPD trajectories 固定下来，在不继续更新参数的条件下，对不同 prefix / visual condition 做 replay，并读取 Teacher / Student 的分布或候选答案 likelihood。

3. 数据采集的第一步：先收集“失败轨迹”

核心 probe 不是从所有成功样本开始，而是从 Student 已经失败的 trajectory 开始。

为什么？

因为作者想研究的问题就是：

当 Student 已经做出了错误视觉判断，而且这个错误逐渐写进语言上下文以后，Teacher 是否还能够纠正它？

因此一个合理的数据记录单元至少需要保存：

sample_id
question x
original image I
privileged crop I+
student full response y
ground-truth answer z*
student realized wrong answer z_err

如果要做 token-level hidden-response probe，还需要能够在每个 visited state (c_t) 上重新获得：

Student distribution          pS_t
Teacher(real evidence)        p+_t
Teacher(null evidence)        p0_t

重要的是：

不是让 Teacher 自己重新生成一条全新 trajectory。

作者把 Student 已经走过的错误 trajectory 当作固定路径，对同一个状态做 counterfactual evaluation。

4. Probe A：Prefix Dominance —— 错误 prefix 会不会把 Teacher 带跑？

这是 Figure 1(a) 对应的 probe，也是发现 Challenge 的第一步。

4.1 研究问题

作者首先不是问：

“Teacher 比 Student 强多少？”

而是问：

如果我只保留越来越多 Student 已经写错的 reasoning，Teacher 在 privileged visual evidence 不变的情况下，还能不能救回来？

如果 Teacher 的 privileged image 真能压过语言上下文，那么无论保留多少错误 reasoning，它都应该仍然倾向正确答案。

如果不是，就会看到：

[
\text{prefix 越长}
\Rightarrow
\text{Teacher correction 越弱}
]

5. “保留 30%、50%、90% 的失败 reasoning”到底是什么意思？

论文定义 retention ratio：

[
\rho
]

并把保留下来的错误 prefix 记作：

[
h_\rho
]

Figure 1 使用：

[
\rho\in{0%,10%,20%,\ldots,90%}
]

概念上，假设一条失败 response 的 reasoning 部分共有 100 个 token：

token 1  ------------------------------------------------------ token 100
|------------------- Student 的失败 reasoning -------------------|

那么：

(\rho=0%)

不保留 Student 的错误 reasoning：

[Question]
[common answer prompt]

Teacher 基本只面对 question + privileged evidence。

(\rho=30%)

保留失败 reasoning 的前 30%：

[Question]
[y1 ... y30]
[common answer prompt]

Teacher 被迫在：

Student 已经讲了 30% 错误故事

的状态下回答。

(\rho=50%)

[Question]
[y1 ... y50]
[common answer prompt]

此时语言上下文里已经积累了更多错误 commitment。

(\rho=90%)

[Question]
[y1 ... y90]
[common answer prompt]

这时 Teacher 已经继承了几乎整条错误推理，只差最后答案附近的部分。

注意：论文写的是“retained prefix followed by a common answer prompt”，也就是每个 retention ratio 后面都接相同的 answer prompt，让不同 (\rho) 下的答案 likelihood 可以在类似回答位置比较。

6. 一个更具体的保留示例

下面是为了理解 retention 而构造的教学例子，不是论文中的真实 benchmark 样本。

假设真实图像中的 pattern 是：

striped

但 Student 失败 response 是：

The pattern appears to be checkered.
I can see alternating square-like regions.
This is characteristic of a checkered texture.
Therefore the object should be classified as checkered.
The best choice is B: checkered.

6.1 0% retained

给 Teacher：

Question: What is the pattern?
[privileged crop: very clear striped pattern]

Answer:

Teacher 没有继承错误故事，视觉证据最容易起作用。

6.2 约 30% retained

给 Teacher：

Question: What is the pattern?
The pattern appears to be checkered.
I can see alternating square-like ...

Answer:

现在视觉 crop 说：

striped

但是 prefix 已经开始说：

checkered

Teacher 开始面对 vision-language conflict。

6.3 约 60% retained

Question: What is the pattern?
The pattern appears to be checkered.
I can see alternating square-like regions.
This is characteristic of a checkered texture.

Answer:

语言上下文已经不是一个孤立错误 token，而是一个逐渐自洽的解释链。

6.4 约 90% retained

Question: What is the pattern?
The pattern appears to be checkered.
I can see alternating square-like regions.
This is characteristic of a checkered texture.
Therefore the object should be classified as checkered.

Answer:

此时如果 Teacher 继续输出：

B: checkered

并不代表它完全没有看见 striped evidence。

它可能只是：

[
\text{prefix-induced continuation pressure}



\text{visual correction}
]

Probe A 就是先证明这件事真的发生。

7. Probe A 如何测量 Teacher 的 decision？

作者不仅看自由生成结果，还定义了一个更稳定的候选答案打分。

对于一个候选答案序列：

[
z=(z_1,\ldots,z_{|z|})
]

计算 length-normalized log-likelihood：

\frac{1}{|z|}
\sum_{k=1}^{|z|}
\log
p_{\phi_P}
(z_k\mid o^+,h_\rho,z_{<k})
]

为什么除以 (|z|)？

因为正确答案和错误答案可能 token 长度不同。
直接比较总 log-likelihood 会天然惩罚更长的 sequence。

7.1 正确 vs 实际错误答案

作者拿两个 candidate：

(z^\star)：ground-truth correct answer

(z^{err})：Student trajectory 最后实际给出的 wrong answer

定义 margin：

\ell(z^\star\mid o^+,h_\rho)

\ell(z^{err}\mid o^+,h_\rho)
]

解释：

(m_\rho>0)

Teacher 在这个 retained prefix 下：

更偏向正确答案。

(m_\rho<0)

Teacher：

更偏向 Student 原来走到的错误答案。

因此最关键的不是 margin 绝对值，而是：

[
m_\rho
\text{ 会不会随着 }\rho\text{ 增大而过零？}
]

8. 作者做了两种 answer preference 对比

Figure 1(a) 下半部分有两行：

1. Answer letter

只比较 multiple-choice letter，例如：

A
vs
B

这样减少选项文本长度和 wording 的干扰。

2. Full answer

比较：

A: striped
vs
B: checkered

也就是 letter + option sequence。

两种指标都从正值变成负值，说明现象不是某一种答案表达格式造成的。

9. Probe A 还按视觉问题类型拆开

Recovery rate 分为：

Overall

Attribute / Object

OCR / Number

Spatial

这样是为了排除：

“prefix dominance 只发生在某一种视觉能力，比如 OCR。”

结果三个类别都随 retention ratio 增大而明显下降。

论文 Figure 1(a) 报告的 recovery：

retained

Overall

Attr/Object

OCR/Number

Spatial

0%

75.8

74.4

76.8

79.2

10%

69.7

71.3

65.7

73.3

20%

67.5

65.2

68.6

74.2

30%

59.5

55.9

63.8

63.3

40%

48.1

44.3

53.3

50.0

50%

41.2

34.3

51.1

43.3

60%

31.7

28.0

36.8

33.3

70%

22.2

18.9

26.7

24.2

80%

10.4

7.5

13.3

14.2

90%

6.0

5.7

8.3

1.7

整体从：

[
75.8%\rightarrow 6.0%
]

Spatial 甚至从：

[
79.2%\rightarrow 1.7%
]

10. 更关键：Teacher preference 真的发生了符号翻转

Figure 1(a) 的 aggregate answer-letter margin：

retained

margin

0%

+3.4

10%

+2.6

20%

+2.3

30%

+1.9

40%

+0.9

50%

+0.0

60%

-0.7

70%

-1.6

80%

-2.9

90%

-3.4

full-answer margin：

retained

margin

0%

+1.0

10%

+0.8

20%

+0.7

30%

+0.5

40%

+0.2

50%

-0.0

60%

-0.3

70%

-0.5

80%

-0.9

90%

-1.0

这不是：

Teacher 只是“没那么有把握”。

而是：

[
\boxed{
\text{Teacher conditional decision 从 correct 翻成 wrong}
}
]

这就是第一个 Challenge 的实验证据。

11. Probe A 的控制变量是什么？

这个实验最重要的设计点是：

保持不变

同一个 question；

同一条 failed Student trajectory；

同一个 privileged visual observation；

同一个 Teacher；

同一对 correct / realized-wrong candidate answer；

同一个 answer prompt。

改变

只有：

[
\boxed{\text{保留多少 Student 的错误 reasoning}}
]

因此观察到的 preference flip 很自然地指向：

accumulated linguistic context 会逐渐覆盖 Teacher 的 visual advantage。

12. Probe A 得到的 Challenge，但还没有得到最终 Finding

Probe A 只能说明：

[
\text{错误 prefix 变长}
\Rightarrow
\text{Teacher 显式 correction 变弱}
]

但这时候有两个可能解释：

解释 1

视觉 evidence 真的已经对 Teacher 完全没用了。

解释 2

视觉 evidence 仍在改变 Teacher，只是这个影响不够强，无法反映到最终 top-level prediction / correct-vs-wrong decision 上。

这两种解释非常不一样。

如果是解释 1：

没什么可以恢复，方法空间很小。

如果是解释 2：

还有一个隐藏 correction 可以被 probe 出来并重新利用。

因此作者继续做 Probe B。

13. Probe B：Hidden Evidence Response —— Teacher 被带跑后，image 还在影响它吗？

Probe B 的设计原则是：

不要再比较两个不同模型；固定同一个 Teacher，只改变它有没有视觉 evidence。

这是整篇论文最重要的 diagnostic intervention。

14. 对同一个 Student visited state，采集三个 distribution

对 failed trajectory 的某个位置 (t)，固定：

[
c_t=(x,y_{<t})
]

然后采集：

14.1 Student distribution

Student 看 full image：

p_\theta(v\mid I,c_t)
]

14.2 Teacher + real privileged evidence

Teacher 看 localized privileged crop：

p_\phi(v\mid I^+,c_t)
]

14.3 同一个 Teacher + visual null

把 (I^+) 的 visual content 去掉，但尺寸保持一致。

论文主方法使用：

mean RGB image

即把 crop 中所有 pixel 替换成该 crop 的平均 RGB 颜色。

得到：

[
I^0=N(I^+)
]

再跑：

p_\phi(v\mid I^0,c_t)
]

15. Probe B 为什么比 Teacher–Student 比较更干净？

比较：

[
p_t^+ \quad\text{vs}\quad p_t^S
]

时，有很多东西同时不同：

model identity；

Student / Teacher 参数；

calibration；

language prior；

optimization state；

visual input。

所以：

[
p_t^+ - p_t^S
]

不是纯 visual effect。

而比较：

[
p_t^+ \quad\text{vs}\quad p_t^0
]

时：

完全相同

Teacher 参数；

question；

Student-generated prefix；

token position；

visual input shape。

唯一核心变化

real privileged evidence

vs evidence-absent visual null

所以：

[
p_t^+ - p_t^0
]

更接近：

视觉 evidence 本身对这个 Teacher token preference 的响应。

16. 第一组量：Standard OPD 能看到多少 correction？

作者用 vocabulary 上的 total variation distance：

\frac12
\sum_{v\in V}
|p_t^+(v)-p_t^S(v)|
]

解释：

privileged Teacher 和 Student 在这个 visited state 上有多不一样。

如果：

[
g_t^S\approx 0
]

说明：

Teacher 和 Student 的 next-token distribution 已经非常接近。

标准 OPD 就会认为：

“这里几乎没有 distribution discrepancy 可以 distill。”

17. 第二组量：视觉 evidence 实际还能让 Teacher 动多少？

同样定义：

\frac12
\sum_{v\in V}
|p_t^+(v)-p_t^0(v)|
]

它测的是：

在完全相同 prefix 下，给 Teacher real visual evidence 相比不给 evidence，prediction 改变了多少。

18. 关键量：Excess visual response

作者定义：

[
e_t=g_t^0-g_t^S
]

如果：

[
e_t>0
]

表示：

同一个 Teacher 对“有没有视觉证据”的 response，仍然比 standard OPD 能从 Teacher–Student gap 中观察到的 correction 更大。

直觉例子：

Teacher vs Student gap           gS = 0.03
Teacher real vs null gap         g0 = 0.12

那么：

[
e=0.12-0.03=0.09
]

这意味着：

Teacher 和 Student 表面上已经很像，但视觉 evidence 仍然能让 Teacher distribution 明显变化。

19. Figure 1(b) 上半图怎么采集和画？

对 failed trajectories 的 visited states，作者收集很多：

[
(g_t^S,e_t)
]

pair。

然后看：

当 (g_t^S) 从较大值逐渐走向 0，也就是 Teacher / Student 越来越 agreement 时，(e_t) 是否也同时变成 0？

如果变成 0：

说明视觉 evidence 的作用真的消失了。

如果仍然 (>0)：

说明 agreement 隐藏了 visual response。

Figure 1(b) 上半图显示：

Teacher–Student 越接近，real-vs-null 的 response 仍然可以保留。

因此：

[
\boxed{
p_t^+\approx p_t^S
\not\Rightarrow
p_t^+\approx p_t^0
}
]

这就是 Aha 的第一半。

20. 但“distribution 还会动”不等于“这个 movement 是正确的”

这里作者又做了一步非常关键的 probe。

因为即使：

[
p_t^+\neq p_t^0
]

也只能说明：

image 改变了 prediction。

还不能说明：

image 在把 prediction 往正确方向推。

于是作者定义 directional score。

21. Directional probe：视觉 evidence 到底更支持 correct 还是 wrong？

对 correct answer (z^\star)：

\log p_t^+(z^\star)

\log p_t^0(z^\star)
]

表示：

加入真实视觉 evidence 后，Teacher 对正确答案的 support 增加多少。

对 Student 实际 wrong answer (z^{err})：

\log p_t^+(z^{err})

\log p_t^0(z^{err})
]

然后：

\Delta^\star_t-\Delta^{err}_t
]

也就是：

[\log p_t^+(z^{err})-\log p_t^0(z^{err})]
]

21.1 解释 (d_t)

(d_t>0)

real visual evidence 相比 visual null：

更加强了 correct answer，而不是 wrong answer。

即视觉响应有 corrective direction。

(d_t<0)

反过来，视觉 evidence 更支持 wrong answer。

22. Figure 1(b) 下半图怎么画？

作者沿着 failed trajectory 的 normalized position：

0% ------------------------------- 100%
early                                 late

在不同位置计算 (d_t)。

想看的问题是：

即使到了 trajectory 很后面、Teacher 的显式 decision 已经被 prefix 拉错，real-vs-null 的视觉 response 是否仍然相对 favor ground truth？

结果：

mean-color counterfactual 和 mismatched-image control 都整体 favor correct direction。

所以更准确的 finding 是：

[
\boxed{
\text{错误 prefix 压制的是 correction 的“显式表达”，而不是完全擦除 visual preference。}
}
]

23. Probe B Control 1：换一个完全不同的 null construction

有人可能质疑：

mean RGB 这种奇怪的人工图，会不会自己产生 artifact？

于是作者把 visual null 替换成：

另一个问题的 mismatched natural image

这时仍然：

same Teacher

same failed prefix

same current state

只是把 visual reference 换成 unrelated natural image。

结果仍然看到 hidden evidence response。

因此 finding 并不依赖某一个特定 mean-color construction。

24. Probe B Control 2：Token-shuffled response

这是非常关键的 control。

假设 real-vs-null 对 token 的影响原来是：

striped       +0.8
checkered     -0.2
wait          +0.3
therefore     -0.4
...

作者把“response 的强度”保留，但把这些变化对应到哪个 token 上打乱。

概念上变成：

striped       -0.4
checkered     +0.3
wait          -0.2
therefore     +0.8
...

注意：

总体 distribution change 的 magnitude 仍然可以很大。

但是：

它已经不再和正确/错误 token 的语义 identity 对齐。

如果原本的效果只是“视觉输入让 distribution 随便动了一下”，那么 shuffle 后应该差不多。

结果不是。

token shuffle 会破坏 corrective direction。

所以作者得到更精确的结论：

真正有价值的是 visual evidence “具体支持了哪些 token”，而不是 distribution change 有多大。

这非常重要，因为 OPD-Aha 后面重建的正是这个 token-specific preference direction。

25. 从 Probe A 到 Probe B：Aha 是怎样一步一步产生的？

可以把作者的推理链写成：

Step 1

在 failed Student trajectory 上逐渐增加 retained prefix：

[
\rho\uparrow
]

观察：

[
Teacher\ recovery\downarrow
]

且：

[
m_\rho:+\rightarrow-
]

所以：

Teacher 会被错误语言 continuation 带跑。

Step 2

Teacher 和 Student 后期越来越相似。

传统解释可能是：

“Teacher 已经没额外知识了。”

Step 3

固定 Teacher 和 prefix，只切换：

real privileged crop
vs
visual null

发现：

[
g_t^0>0
]

甚至当：

[
g_t^S\approx0
]

时仍存在。

所以：

visual effect 没消失，只是 Teacher–Student gap 看不见。

Step 4

再看方向：

[
d_t>0
]

说明 real visual evidence 仍然相对 favor correct answer。

Step 5

token shuffle 后这个方向消失。

所以：

hidden signal 不是“任意 perturbation”，而是一个 token-specific corrective visual preference。

这一步之后，方法几乎自然出现：

[
u_t(v)=
\log p_t^+(v)-\log p_t^0(v)
]

既然这是被隐藏的 visual preference，那就把它重新写进 distillation target。

26. 一个完整 toy example：从 failed trajectory 到 hidden visual signal

下面仍然是教学例子。

26.1 Student 已经失败

图片真实 pattern：

striped

Student full response：

The pattern looks checkered.
The alternating regions form square-like cells.
So this should be a checkered pattern.
Answer: B.

Ground truth：

A: striped

Realized wrong：

B: checkered

26.2 在短 prefix 处

Teacher + privileged crop：

P(A: striped)   = high
P(B: checkered) = low

因此：

[
m_\rho>0
]

26.3 保留越来越长的错误 prefix

到后面：

The pattern looks checkered.
The alternating regions form square-like cells.
So this should be a checkered pattern.
...

即使 crop 很清楚，Teacher 也可能变成：

P(A: striped)   = 0.30
P(B: checkered) = 0.55

所以：

[
m_\rho<0
]

Probe A 到这里说明 Teacher 已经被带跑。

26.4 现在不要和 Student 比，改成同一个 Teacher 的 real/null 对比

visual null 下：

P(A: striped)   = 0.10
P(B: checkered) = 0.70

real crop 下：

P(A: striped)   = 0.30
P(B: checkered) = 0.55

注意：

real crop 下 Teacher 仍然最终偏 wrong：

[
0.55>0.30
]

但视觉 evidence 的效果其实是：

striped    0.10 -> 0.30   大幅增加
checkered  0.70 -> 0.55   减少

也就是说：

Teacher 的最终 decision 错了，但 image 对它的局部 preference change 是对的。

这正是 hidden correction。

27. Probe C：Reconstructed target 是否真的能在长错误 prefix 上恢复 supervision？

前两个 probe 是为了“发现 finding”。

有了方法之后，作者继续 probe：

reconstructed target (q_t) 是否真的比标准 privileged target (p_t^+) 更能抵抗 prefix dominance？

仍然从 failed Student responses 出发，逐渐保留更多 continuation。

对同一个 state 比较：

Standard privileged target

[
p_t^+
]

Reconstructed target

[
q_t
]

看 correct-answer probability。

结果：

早期 prefix：两者接近；

prefix 越长：(p_t^+) 的 correct-answer probability 快速下降；

(q_t) 保持明显更高；

late-prefix state 的 relative amplification 超过一个数量级，Figure 3 中出现约 (14.6\times)、(17.2\times) 的放大。

这验证：

real-null residual 不是只有统计意义；把它写回 target 后，确实可以恢复 standard target 丢失的 late-prefix correction。

28. Probe D：为什么会自动出现 wait / actually？

作者注意到一个现象：

训练没有标注 wait、actually，但 reconstruction 强时这些 reflection word 会变多。

于是他们专门 probe。

28.1 数据怎么采集？

从 Student trajectories 中找到：

马上要出现第一个 reflection word 的 prefix

例如：

... current wrong reasoning [PREFIX] -> "wait"

然后找相似的 control：

在相同 response 位置附近，但 trajectory 不进入 reflection。

论文描述为 similar / matched positions。

28.2 对每个 prefix 测三个量

比较 reconstruction 前后：

A. Reflection word 相对当前 continuation 的 preference

例如：

[
\log
\frac{q_t(\text{wait})}
{q_t(\text{continue-word})}
]

相对 standard target 发生多大变化。

B. Reflection word 的 absolute probability

[
q_t(\text{wait})
]

到底有没有直接升高。

C. Correct answer 的 absolute probability

[
q_t(z^\star)
]

是不是已经被直接 boost。

28.3 结果

作者发现：

wait / actually 相对 ongoing continuation 变得更有竞争力；

但它们的 absolute probability 可以下降；

correct answer 的 absolute probability 也没有在这个时刻被直接提高。

因此真正发生的是：

[
\text{wrong continuation 被压得更多}
]

而不是：

[
\text{wait 被硬推高}
]

所以 wait 成为最自然的“退出当前 reasoning branch”的语言 token。

29. Frozen checkpoint probe：模型是不是只学会一种“反思文风”？

作者又保存训练中的 frozen checkpoints：

Base
step 10
step 20
step 30
step 40

在：

真正 reflection 前的 prefix；

matched non-reflection position；

分别测 reflection-word preference。

结果是：

early training

reflection words 在很多位置都上升。

later training

这种 broad increase 被逐渐删掉：

matched non-reflection position 下降；

真正需要 reflection 的 prefix 保留。

因此 Student 不是简单学成：

“凡事都说 wait。”

而是逐渐学成：

在特定 image-language conflict state 下使用 reflection。

30. Probe E：说了 wait 以后真的重新看图了吗？

光看到 reflection words 还不够。

可能只是 stylistic behavior：

wait... actually...

但模型仍然继续依赖文字惯性。

于是作者继续做 evidence-source probe。

30.1 数据分组

找到含有 reflection token 的 response。

以：

第一个 reflection token

为 anchor。

再找：

没有 reflection token、但在相同 response position 对齐，并按 benchmark 和 final correctness 匹配的 control responses。

这个 matching 很重要，因为否则：

reflection response 可能天然更难；

或者出现位置不同；

或者最终正确率不同。

30.2 Reflection 后分窗口

统计：

1--8 tokens
9--16 tokens
17--32 tokens
33--64 tokens

30.3 测两个 evidence source

Continuation support

已有 accumulated text 对后续生成的支持。

Visual-evidence support

image 对后续 token 的支持。

30.4 结果不是“wait 后立刻看图”

前 1--8 token：

两组 separation 不大。

然后：

continuation support 逐渐下降；

visual-evidence support 逐渐上升；

最明显的 separation 出现在 reflection 后约 17 tokens 以后。

因此行为顺序是：

[
\boxed{
reflection
\rightarrow
break linguistic commitment
\rightarrow
deliberation
\rightarrow
visual evidence re-enters continuation
\rightarrow
answer revision
}
]

而不是：

[
wait\rightarrow correct\ answer
]

31. Probe F：隐藏 visual correction 能活到 trajectory 多后面？

Appendix Figure 7 把 failed trajectory 归一化分为：

Q1：earliest

Q2

Q3

Q4：latest

然后问：

对一个已经偏 wrong 的 Teacher state，reconstruction 能否用 visual residual 把 preference 翻回来？

随着 (\beta) 增加，统计：

hidden corrections reversed (%)

在较强 reconstruction 时：

Q1：64.6%

Q2：54.4%

Q3：49.9%

Q4：42.1%

并且需要的 median (\beta_{\min}) 从：

[
2.24\ (Q1)
]

增长到：

[
5.55\ (Q4)
]

解释：

visual correction 后期仍然存在，但错误 prefix 越深，语言惯性越强，需要更大的 reconstruction strength 才能把正确 token 翻回来。

32. Probe G：是不是只要把 target 改得够大就有效？

另一个可能的替代解释：

OPD-Aha 有效，也许不是 visual direction 对，而只是它把 target 改动得更大。

作者做 token shuffle control，并尽量保持：

Control 1

similar total target change

Control 2

similar target uncertainty

但把：

哪个 token 获得哪个 visual change

打乱。

结果 observed visual residual 相比 shuffled controls 仍然有明显 correct-option margin gain：

约 0.396

约 0.379

所以：

有用的不是“target 被扰动了多少”，而是“视觉 evidence 把 mass 往哪些 token 搬”。

33. Probe H：这个 correction 真来自 task-relevant crop 吗？

再一个替代解释：

也许只是有一个额外 crop / extra image input 就会产生效果。

作者在同一张 image 内构造：

Relevant

原 task-relevant localized crop

Irrelevant control

与 relevant crop：

geometry matched；

但 spatially non-overlapping；

来自同一原图的无关区域。

这样可以减少：

图片风格；

场景；

分辨率；

输入尺寸；

等 confound。

在 (\beta=4) 的 late hidden corrections：

task-relevant crop：48.4%

irrelevant-region crop：39.4%

差值：

[
+9.1\text{ pp}
]

95% CI：

[
[2.9,15.1]
]

因此 correction 的强度确实和：

问题相关的视觉区域

相关。

34. 从“数据采集”角度重新写成完整 pipeline

如果你要复现论文的 probing 思路，可以把流程理解为下面几步。

Stage 1：生成并保存 on-policy trajectories

对每个 prompt：

for prompt in dataset:
    y = student.generate(original_image, question)
    save(y)

根据 final answer 判断：

if final_answer(y) != ground_truth:
    failed_pool.append(...)

论文明确说核心 probe 使用 failed Vision-OPD trajectories；
但当前 draft 没有写 Figure 1 的精确样本数量和过滤实现。

Stage 2：Prefix-retention replay

对每条 failed trajectory：

for rho in [0.0, 0.1, ..., 0.9]:
    retained = first_rho_fraction_of_failed_reasoning(y, rho)
    h_rho = retained + common_answer_prompt

    score_correct = teacher.sequence_logprob(
        privileged_crop,
        question,
        h_rho,
        correct_answer,
    )

    score_wrong = teacher.sequence_logprob(
        privileged_crop,
        question,
        h_rho,
        realized_wrong_answer,
    )

    margin = normalized(score_correct) - normalized(score_wrong)

聚合：

rho -> recovery %
rho -> answer-letter margin
rho -> full-answer margin
rho -> category-specific recovery

Stage 3：Visited-state distribution replay

对 failed trajectory 的每个 token position：

prefix = y[:t]

pS = student.next_token_distribution(
    original_image,
    question,
    prefix,
)

p_plus = teacher.next_token_distribution(
    privileged_crop,
    question,
    prefix,
)

p_zero = teacher.next_token_distribution(
    mean_rgb_crop,
    question,
    prefix,
)

这里最重要的是：

question 相同
prefix 相同
token position 相同
Teacher 参数相同

real / null 只改变 evidence。

Stage 4：计算 hidden response

gS = 0.5 * L1(p_plus, pS)
g0 = 0.5 * L1(p_plus, p_zero)
e  = g0 - gS

记录：

records.append({
    "position": t / len(y),
    "g_student": gS,
    "g_visual": g0,
    "excess_visual_response": e,
})

Stage 5：检查 correction direction

delta_correct = (
    logprob(p_plus, correct_answer)
    - logprob(p_zero, correct_answer)
)

delta_wrong = (
    logprob(p_plus, wrong_answer)
    - logprob(p_zero, wrong_answer)
)

d = delta_correct - delta_wrong

如果：

d > 0

说明：

real evidence 相比 null 更向 ground truth 倾斜。

Stage 6：Counterfactual controls

Mismatched image

p_mismatch = teacher(
    unrelated_natural_image,
    same_question,
    same_prefix,
)

检查 finding 是否仍成立。

Token shuffle

对 visual response 在 vocabulary token identity 上置乱，再重新看 directional correction。

目标是区分：

“distribution 变化”

和：

“正确 token 得到正确方向的变化”

35. 这套 probe 最漂亮的实验设计点

我认为有四个。

35.1 不让 Teacher 自己重新 rollout

如果让 Teacher 从头生成，它很可能一开始就走到另一条正确 branch。

这样你测到的是：

Teacher 独立生成能力更强。

但论文关心的是：

Teacher 在 Student 真正访问的错误 state 上还能不能纠错？

所以必须固定 Student trajectory。

35.2 Prefix retention 是一种“剂量实验”

[
\rho=0,0.1,\ldots,0.9
]

像逐步增加 treatment dose。

其他条件不变，只提高：

已经积累的错误 language context 数量。

因此可以看到一个清晰的 monotonic degradation，而不是只比较“短 prefix vs 长 prefix”两个点。

35.3 Real-null comparison 是 within-model paired intervention

不是：

Teacher A vs Student B

而是：

same Teacher
same language prefix
same state
real evidence vs null evidence

所以它把 model identity 这个巨大的 confound 消掉了。

35.4 Token-shuffle 是 directionality control

作者不满足于证明：

image 改变了 distribution。

而是继续证明：

image 改变的是“正确的 token identity”。

这让 finding 从：

visual input still has influence

提升成：

visual input still carries a recoverable corrective direction

后者才能真正支撑 OPD-Aha。

36. Probe 的核心因果图

可以把变量关系想成：

Privileged visual evidence --------------------+
                                                |
                                                v
                                        Teacher token preference
                                                ^
                                                |
Student-generated erroneous prefix ------------+

当 prefix 很短：

visual evidence influence > prefix inertia

Teacher 显式纠错。

当 prefix 很长：

prefix inertia > visual evidence influence

Teacher 的最终 prediction 被拉错。

但 Probe B 通过：

same Teacher(real evidence)
-
same Teacher(null evidence)

把 visual evidence 这一条边单独测出来。

因此最终发现：

Teacher final prediction wrong

并不能推出：

visual evidence has zero corrective influence

37. 最重要的一句话：Probe 真正改变了作者如何“看这个问题”

没有 Probe B 时，问题看起来是：

“错误 prefix 已经让 privileged Teacher 也错了，所以 Teacher 没法提供监督。”

这是一个近乎无解的表述。

Probe B 后，问题变成：

“错误 prefix 只是让 visual correction 无法在 Teacher 的最终 prediction 中占上风，但 same-Teacher counterfactual difference 仍然把这个 correction 暴露出来。”

于是问题变成可解：

[
\boxed{
\text{不是重新创造 correction，而是恢复一个仍然存在但被压制的 correction。}
}
]

这就是这篇论文真正的 Aha finding。

38. 论文当前没有交代、复现时最好向作者确认的 Probe 细节

这部分非常重要，因为当前版本是 under-review draft。

Figure 1 的核心 probe 对下列细节没有完整说明：

Figure 1 的 Student / Teacher 是 4B 还是 9B；

使用哪个 Vision-OPD checkpoint；

failed trajectories 的准确样本数；

每个 Attr/Object、OCR/Number、Spatial 类别分别多少样本；

failed pool 来自哪些 benchmark / split；

common answer prompt 的精确 template；

reasoning span 如何和最终 answer span 分离；

retention token 数如何取整；

recovery (%) 的精确 decoding / 判定方式；

(d_t) 中 answer sequence log-prob 的具体 tokenization / normalization 细节；

token shuffle 的具体 permutation 粒度和随机重复次数；

Figure 1(b) 上曲线的 binning、smoothing 和置信区间实现。

因此，如果真正做 reproduction，推荐把这些当成一个 checklist，而不是自行假设。

39. 最后用一张“Probe → 结论”表总结

Probe

固定什么

改变什么

测什么

得到什么结论

Prefix retention

Teacher、privileged image、failed trajectory、answer candidates

retained wrong prefix 比例

recovery、(m_\rho)

错误 prefix 越长，Teacher 越被带跑

Teacher–Student gap

visited state

模型/视觉输入不同

(g_t^S)

standard OPD 能看到的 correction 会收缩

Real vs null

same Teacher、same prefix

visual evidence present/absent

(g_t^0)

即使 T/S agreement，visual response 仍存在

Directional response

same state

real vs null

(d_t)

hidden response 倾向 correct answer

Mismatched image

same Teacher/prefix

null construction

residual direction

finding 不依赖 mean-color 特例

Token shuffle

response magnitude

token identity alignment

correct-option direction

有用的是 token-specific visual direction

Reconstructed target

same failed retained state

(p_t^+) vs (q_t)

correct-answer prob

residual 可以恢复 late-prefix supervision

Reflection prefix

matched state

是否将进入 reflection

relative/absolute token preference

reflection 来自削弱 wrong continuation

Post-reflection evidence

matched position/correctness

reflection vs no reflection

text / visual support

reflection 后逐渐重新依赖 image

Relevant-region control

same image / geometry

relevant vs non-overlap crop

hidden correction reversal

correction 指向 task-relevant region

40. 一句话复现思路

如果只记一个 implementation recipe：

1. 先从 Vision-OPD 收集最终答错的 Student trajectories；
2. 固定这些 trajectories，不让 Teacher 自己重新 rollout；
3. 逐步保留 0%~90% 的错误 reasoning，测 Teacher correct-vs-wrong margin；
4. 在每个 Student visited state 上同时记录 pS、p+、p0；
5. 检查 Teacher–Student gap 是否消失，但 real-null gap 是否仍存在；
6. 再检查 real-null residual 是否 specific 地 favor ground truth；
7. 用 mismatched image 和 token shuffle 排除 artifact / generic perturbation；
8. 只有在这些 probe 成立后，才有理由把 log p+ - log p0 当成 visual correction 并重构 distillation target。

这条链条就是：

[
\boxed{
\text{Challenge probe}
\rightarrow
\text{hidden-signal probe}
\rightarrow
\text{directionality control}
\rightarrow
\text{Aha finding}
\rightarrow
\text{method}
}
]