# 晶體擴散生成與物理引導搜索研究計畫

> 以 Diffusion 提供多樣化起點，透過 MD／退火探索能量盆地，再以無偏置弛豫、300 K MD 與 DFT 分層驗證。

## 計畫概覽

| 項目 | 規劃 |
| --- | --- |
| 研究週期 | 12 週（三個月） |
| 研究對象 | 無機週期晶體；MD 軌跡內固定組成；優先採用約 5–30 原子／原胞的體系 |
| 首個目標性質 | DFT 定義一致的 band gap；體積模量等作為擴展條件 |
| 核心比較 | 無偏置 MD、連續 Property-biased MD、Basin-level Guidance |
| 驗證流程 | 無偏置 0 K 弛豫 → 300 K MD／DFT 分層驗證 |
| 決策節點 | W3：搜索收益；W7：回流樣本品質；W7–W8 決定是否啟動微調 |
| 回流策略 | 物理驗證通過後，視條件開展 Predictor 主動學習及 1–2 輪 Diffusion 微調 |
| 範圍限制 | 不從零訓練生成模型；暫不研究 Flow Matching 蒸餾 |

## 摘要

晶體擴散生成模型已能提出大量幾何合理且接近局部平衡態的候選結構，但「預測性質達標」並不等同於生成了真實穩定、性質符合要求的新材料。另一方面，直接使用性質預測網路的梯度優化晶格與原子位置，可能停留在原有能量盆地，或利用代理模型在非平衡幾何上的誤差。為此，本研究提出以擴散模型提供多樣化初始晶體（粗胚），以有限溫度 MD／模擬退火探索能量盆地，以無偏置 0 K 弛豫確認局部極小值，再以 300 K MD 和 DFT 分層驗證的研究路線。重點比較「無偏置 MD + 弛豫後性質篩選」與「連續施加 property bias 的 MD」，識別物理搜索和代理梯度各自的實際貢獻；只有在前期證明物理驗證收益後，才將可靠候選回流微調擴散生成器，並視資料品質開展性質預測器主動學習。本計畫暫不研究 Flow Matching 蒸餾。週期為 12 週，採漸進式 Go / No-Go 決策，避免把計算資源投入未驗證的高溫偏置策略。

**關鍵詞：** 晶體生成；Diffusion；分子動力學；模擬退火；Minima Hopping；性質偏置勢；主動學習；物理驗證。

## 1. 研究背景與問題定義

### 1.1 研究動機

晶體逆向設計需要同時滿足元素組成、週期幾何、物理局部穩定性及目標性質等約束。CDVAE、DiffCSP 和 MatterGen 等已奠定週期晶體生成的深度學習基礎；MatterGen 亦展示多種材料性質的條件生成，以及穩定、唯一且新穎（SUN）材料的系統性評估[1–3]。然而，對特定功能性質的命中率、生成後弛豫的性質保持，以及對未知原型的真實發現能力，仍不能單靠生成分佈或代理預測器保證。

傳統 Minima Hopping 使用短程 MD 脫離局部極小值，再局部弛豫以尋找新的低能構型[4]；晶胞 metadynamics 及 MLIP 輔助的 melt–quench–anneal 也已展示「熱力學搜索 + 局部弛豫」的可行性[5,6]。預訓練 MLIP（如 CHGNet）使這類模擬較容易在中等規模材料集上開展，但高能態勢能曲面的外推偏差仍須監控[7,10]。

因此，本計畫不是宣稱首次組合 MD 與結構弛豫，而是研究：擴散先驗是否能提升盆地搜索效率？是否應將性質梯度連續施加於 MD，或應將 MD 搜索與弛豫後性質評分解耦？經物理驗證的探索結果又能否回流改善下一輪生成？

### 1.2 研究對象與邊界

主要考察無機、週期、固定組成的晶體，優先選擇約 5–30 原子／原胞、元素化學空間相對可控的體系，後續將選定候選擴展為 2×2×2 等超胞。Diffusion 可跨組成生成初始候選；一旦進入某條 MD 軌跡，元素種類與原子數原則上固定，僅改變原子座標與（在允許時）晶胞。首個目標性質建議採用 DFT 定義一致的 band gap，並保留體積模量等作為擴展條件。首期不處理缺陷濃度與化學反應路徑的變組成問題，也不宣稱 300 K 短程 MD 足以證明可合成性。

## 2. 研究現狀與創新邊界

| 研究線索 | 已有代表工作 | 已明確做到的事 | 本研究仍需實證的差別 |
| --- | --- | --- | --- |
| 晶體擴散生成 | CDVAE、DiffCSP、MatterGen [1–3] | 無條件／條件晶體生成、部分物理驗證 | 是否可作為多盆地搜索起點，超越單純增加生成數量 |
| MD 逃離盆地 + 局部弛豫 | Minima Hopping [4] | 以短程 MD 和 relaxation 搜索新局部極小值 | Diffusion 起點與目標性質驅動的盆地排序能否提高搜索收益 |
| 偏置 MD 與晶相變化 | 晶胞 metadynamics [5] | 利用偏置 collective variables 促進結構轉換 | 用可微 property predictor 建構偏置後，是否保留弛豫後目標性質 |
| 高溫採樣／淬火 | MLIP 輔助 melt–quench–anneal [6] | 高溫軌跡提供候選或訓練資料 | 在相同預算下，生成先驗能否帶來更高品質、更多獨立 basin |
| 性質梯度直接優化 | SMOACS [8] | 代理性質網路梯度可直接優化晶體 | 連續偏置 MD相對弛豫後評分的真實物理收益與失敗機理 |
| 生成—選擇—再訓練 | InvDesFlow-AL [9] | 生成、MLIP 弛豫、代理篩選及微調閉環 | 以盆地多樣性、DFT 獨立驗證及偏置策略消融控制回饋品質 |
| MLIP 高能態可靠性 | Deng 等 [10] | 指出多種通用 MLIP 有系統性的勢能曲面軟化 | 以高能軌跡不確定性與 DFT 抽查設計可信的停止／拒絕條件 |

### 創新邊界聲明

「Diffusion + Relaxation」「MD 脫離局部極小值」「Property-gradient Optimization」「選優樣本後微調生成器」均已有相關先例。

尚待本計畫驗證的貢獻包括：

- 系統比較 property-continuous bias 與 basin-level guidance。
- 揭示偏置後改善有多少能在無偏置弛豫後保留。
- 以相同計算預算比較 diffusion 初始化與傳統搜索。
- 利用經物理驗證且保留原型多樣性的樣本回訓，評估生成模型本身是否改善。

不能預先表述為已證明的新方法。

## 3. 研究假設與核心科學問題

- **RQ1｜探索：**在相同 MLIP 力評估成本下，Diffusion 初始化 + 高溫 MD / 退火能否比 Diffusion + 直接 0 K 弛豫、隨機起點 MD 或僅增加 Diffusion 採樣量，找到更多獨立的、物理上合理的局部極小值？

- **RQ2｜偏置：**在高溫非平衡構型上直接引入 property predictor 的虛擬力，是否真正提高弛豫後目標性質命中率，還是主要造成可逆形變及代理模型利用？

- **RQ3｜穩定：**在無偏置 0 K 弛豫後，有利結構能否在短時 300 K MD 中維持結構特徵？與 DFT 形成能、凸包和必要的聲子檢查是否一致？

- **RQ4｜閉環：**選取經獨立驗證且多樣化的晶體回流 Diffusion 後，新生成器是否在不依賴測試時偏置優化的情況下提升 validated hit rate，而不是只降低代理 loss 或塌縮到少數原型？

### 3.1 可檢驗假設

- H1：相同總 GPU 計算預算下，多起點 MD–Quench–Relax 能提高獨立物理盆地覆蓋率，但增益取決於溫度、胞尺寸與起點多樣性。

- H2：在僅用 equilibrium 晶體標籤訓練 predictor 的設定下，continuous property bias 可能不及「MD 探索 + 弛豫後評分」，這是待測假設而非已知結論。

- H3：嚴格保留 unbiased relaxation、DFT 獨立評估及 diversity selection 的回饋樣本，可以改善下一輪生成品質；是否抵消資料塌縮需消融確認。

## 4. 理論模型與技術路線

### 4.1 晶體表示與起始生成

令候選晶體為

$$
\mathcal C=(A,R,L),\qquad \mathcal C_0\sim p_{\theta}(\mathcal C\mid y^{\ast}),
$$

其中 $A$ 是原子種類，$R$ 是笛卡兒座標，$L$ 是晶胞矩陣，$y^\ast$ 為目標性質。先採用現有預訓練晶體生成器，不從零訓練 diffusion；依任務可選用固定組成的 DiffCSP 或可改變元素種類的 MatterGen。對生成結果進行碰撞檢查、晶胞有效性檢查、去重，並記錄未經優化的基準表現。

### 4.2 物理勢能與 Property Bias

令可微 MLIP 提供 $E_{\mathrm{MLIP}}(R,L;A)$，獨立性質網路預測 $\hat y_q=f_q(A,R,L)$。對多個 property 定義帶尺度歸一的偏置能量：

$$
U_{\mathrm{bias}}(\mathcal C;y^{\ast})
=\frac{1}{2}\sum_{q=1}^{Q}\lambda_q
\left(\frac{f_q(\mathcal C)-y_q^{\ast}}{\sigma_q}\right)^2,
\qquad
U_{\mathrm{eff}}=E_{\mathrm{MLIP}}+U_{\mathrm{bias}},
$$

其中 $\lambda_q$ 的單位為能量，$\sigma_q$ 為對應性質的尺度。與之對應的合力：

$$
\mathbf F_j^{\mathrm{eff}}
=-\nabla_{\mathbf r_j}E_{\mathrm{MLIP}}
-\nabla_{\mathbf r_j}U_{\mathrm{bias}}.
$$

若同時調節晶胞，需要再對晶格應變求導，得到物理應力與 property virtual stress；不能直接把一般座標梯度當作晶格應力。高溫 MD 中 property predictor 可能超出訓練分佈，故連續偏置僅為對照策略，須設定最大原子位移、碰撞拒絕、偏置力上限及模型不確定性門檻。對帶隙等通常在 0 K 結構上標註的性質，熱態瞬時預測值不是室溫實驗帶隙。

### 4.3 三條核心搜索分支

#### 路線 A：Physical MD（無偏置）

對 diffusion 初始結構做短程碰撞消除與 0 K warm-up，之後在 $E_{\mathrm{MLIP}}$ 下執行高溫 MD、退火或短脈衝逃逸，定期保存快照，逐幀進行無偏置結構弛豫；使用獨立性質評分選擇不同局部極小值。這是首選穩健基線。

#### 路線 B：Continuous Property-biased MD

在同樣熱程序下以 $U_{\mathrm{eff}}$ 積分 MD，在軌跡中不斷重新計算當前構型上的 property virtual force；定期抽取快照後去除偏置、做 0 K MLIP relaxation，再評估性質保留。不得把上一時刻虛擬力不加更新地沿用多步，否則不一定是在同一勢能面上積分。

#### 路線 C：Basin-level Guidance

完全由無偏置 MD 產生新盆地，使用弛豫後性質、能量及不確定性為每個 basin 評分；將搜索預算優先分配給前景更好但尚未充分探索的 basin（例如再啟動短程 MD／微擾）。此路線不依賴非平衡性質梯度，是方法設計中的重要對照。

三條分支均應允許多種隨機種子、初始速度、適量晶格擾動與多快照 Quench；不只取 MD 最後一幀。避免將所有樣本在同一小原胞內搜索時的「盆地多樣性」錯誤外推為真實材料相空間完整覆蓋。

### 4.4 0 K 物理弛豫、室溫測試與 DFT

不同偏置策略產生候選後，統一移除 property bias，按相同 MLIP 與力閾值執行局部弛豫：

$$
\mathcal C^{\mathrm{relax}}
=\operatorname*{arg\,min}_{\mathcal C\in\mathcal B(\mathcal C^{\mathrm{snapshot}})}E_{\mathrm{MLIP}}(\mathcal C),
\qquad
\max_j\|\mathbf F_j^{\mathrm{MLIP}}\|<F_{\mathrm{tol}},
$$

$\mathcal B$ 表示快照所在局部吸引盆地；此步不保證找到全局最深極小值。在完成去重、獨立性質及能量篩選後，對少量 top candidates 開展無偏置 300 K MD（如短程 NVT，必要時檢查 NPT 或超胞），監測 RMSD、配位、體積、晶格與不可逆形變；它檢查的是有限模擬時間尺度上的結構持續性，不是熱力學穩定性的證明。進一步以 DFT 弛豫、形成能與可比參考相的 $E_{\mathrm{hull}}$、電子性質等進行高品質驗證，關鍵候選視情況做聲子檢查。需要統一 DFT functional、$+U$、磁態、k 網格和凸包參考資料，避免混用不兼容的能量。

### 4.5 兩個回流分支（條件啟動）

#### 回流 1｜Predictor 主動學習

對模型不確定且代表性好的弛豫結構進行 DFT，更新 property predictor；必要時選取少量受控畸變構型進行性質標註，以檢查幾何梯度。固定獨立 DFT 測試集；報告 MAE 之外的方向導數一致性或擾動排序保真度。

#### 回流 2｜Diffusion 微調

僅使用無偏置弛豫後、通過品質審核且已去重的優質候選回流，與原始真實結構混合訓練，控制新增樣本比例。每輪（建議至多 1–2 輪）固定 test seeds 評估新的 diffusion，在不執行 property-biased MD 的情況下直接生成，再用相同物理驗證流程評估改善程度。InvDesFlow-AL 已存在生成–MLIP–選擇–微調先例[9]，本研究重點放在 bias 策略與 basin 多樣性的回饋價值。

## 5. 評估指標、消融與判定標準

### 5.1 必須報告的主要指標

**獨立驗證的穩定性—性質聯合成功率（主指標）：**

$$
S_{\mathrm{joint}}
=\frac{1}{N}\sum_{i=1}^{N}
\mathbb I\!\left[\left|y_{\mathrm{DFT},i}-y^{\ast}\right|\leq\epsilon_y\right]
\mathbb I\!\left[E_{\mathrm{hull},i}<\delta\right].
$$

若計算預算不允許全部 DFT，需清楚區分 proxy success rate（全樣本）、DFT verified top-M success（受選樣本）及固定 DFT 配額下的成功材料數；不得用 top-M 成功率宣稱全體生成分佈成功率。

**盆地覆蓋與跨盆地效果：**將所有樣本統一無偏置弛豫後，以結構匹配和組成／空間群條件定義 basin proxy，報告 unique minima 數、unique minima / 1,000 MLIP force calls、相較起點進入不同 basin 的比例；此處的結構聚類只是盆地識別的代理，不是嚴格的勢能吸引域定義。

**性質保留：**對同一套獨立 property evaluator，記錄初始物理弛豫誤差 $e_0$、偏置末態誤差 $e_b$、去偏置後誤差 $e_f$；主要報告 $e_f-e_0$ 與 $P(e_f<e_0)$，僅在分母足夠大時附加保留率 $R=(e_0-e_f)/(e_0-e_b)$。

**效率及多樣性：**GPU·h、實際 wall clock、MLIP／property-gradient 調用數、不同原型數、SUN（明確資料庫和閾值）、每 GPU·h 的獨立可用候選數、每固定 DFT 配額的成功數。對比同一批 seed 和相同總預算，提供 bootstrap 信賴區間。

**300 K 魯棒性：**MD 運行期間的 RDF / RMSD、配位、體積與相變事件；標明晶胞大小、系綜、溫控方法和採樣時間，不能據此直接斷言可合成。

### 5.2 對照與消融矩陣

| 組別 | 初始結構 | 搜索 / 弛豫 | 性質參與方式 | 識別目的 |
| --- | --- | --- | --- | --- |
| B0 | Diffusion | 直接 0 K relax | 無 | 生成器本身基準 |
| B1 | Diffusion | 多取樣 + 0 K relax | 僅最終排序 | Best-of-N 成本匹配基準 |
| B2 | Diffusion | 無偏置 MD + 多快照 quench | 弛豫後排序 | 純物理跨盆地探索 |
| B3 | Diffusion | Property-biased MD + quench | 每 MD 步施加虛擬力 | 虛擬力能否帶來額外收益 |
| B4 | Diffusion | 無偏置 MD + 自適應重啟 | basin-level property ranking | 搜索與性質評估解耦 |
| B5 | 隨機／模板 | 與 B2 相同的 MD 預算 | 弛豫後排序 | Diffusion 起點是否真正必要 |
| B6 | Diffusion + 回訓 | 僅 0 K relax（無偏置搜索） | 訓練階段條件 | 回流後生成器本身是否提升 |

其中 B3 與 B2 的差距回答「連續虛擬力是否有必要」；B4 與 B3 的差距回答「對非平衡結構求 property 梯度是否划算」；B5 直接比較 diffusion 初始化和傳統 Minima Hopping 的相對優勢。若 B2/B4 已明顯優於 B3，應將主方法定位為 property-guided basin selection，不勉強主打 continuous bias。

### 5.3 預設決策門檻（規劃值，非先驗結論）

第 3 週以 50–100 個 paired diffusion 起點做試點：若 B2/B3/B4 相對 B0/B1 在無偏置後沒有可重複的聯合成功率或獨立 basin 覆蓋增益，先暫停回訓並改進溫控、超胞、模型校準或探索策略。對 DFT 樣本過少的情況不做顯著性宣稱，優先看效果大小、配對結果和 CI。第 7–8 週只在回流池具備充足物理有效、獨立且多樣化候選時啟動 Diffusion 微調。正式數值門檻應依 pilot 效應量和驗證成本調整。

## 6. 三個月工作計畫（12 週）

| 週次 | 核心任務 | 里程碑與交付 |
| --- | --- | --- |
| W1 | 選材料族與目標性質；配置預訓練 Diffusion、MLIP、Property Predictor、ASE；定義資料切分 | 能重現 B0、B1；建立資料版本與耗時 profiling |
| W2 | 固定 seed 生成 50–100 個候選；溫度／系綜／步長小規模網格；比對 MLIP / DFT 的抽查 | MD 數值穩定與高能態可靠性報告 |
| W3 | B0–B4 的 paired pilot；無偏置弛豫、盆地去重與 proxy 性質評估 | Go / No-Go 1：MD 與偏置是否真能提升有效 basin 搜索 |
| W4 | 擴大多起點試驗，系統比較高溫 pulse、退火、快照間隔與擾動強度 | 搜索成本–品質 Pareto 曲線 |
| W5 | Continuous bias 與 basin-level ranking 完整消融；固定 DFT Top-M 配額 | 確認主方案與 surrogate failure modes |
| W6 | 首輪 DFT 弛豫、形成能與性質驗證；選取 300 K MD 名單 | Independent validated hit rate、保留率與 benchmark |
| W7 | 300 K NVT / 適量超胞測試；必要時做 predictor 校準或小量主動學習 | Go / No-Go 2：是否具備可回流的高品質、非重複樣本 |
| W8 | 以原始資料混合回流樣本微調 diffusion，並保留未微調 checkpoint | 第一輪 Diffusion feedback ablation |
| W9 | 微調模型重新生成與無偏置評估；是否開展第二輪回流 | 不依賴 TTO 的生成品質改善曲線 |
| W10 | 固定採樣數與 GPU/DFT 配額完成對照；可選第二輪回訓 | 搜索策略與閉環策略拆分消融 |
| W11 | 獨立 DFT 複核；關鍵案例進行 300 K MD / 聲子等補充驗證 | 不確定性、穩定性及新穎性完整評估 |
| W12 | 統計、消融、失敗案例與成本核算；完成論文初稿 | 開題後階段總結／論文草稿／可重現程式 |

## 7. 計算資源與成本估算

### 7.1 估算口徑與基準假設

計算資源對 MLIP 類型、原子數、鄰居數、運行精度、硬體、MD 系綜以及是否每步反向傳播 property predictor 極度敏感。下面是編製計畫用的假設區間，並非對特定框架的實測 benchmark；第一週必須對目標機器 profiling 後修正。

建議核心設定：單個原胞約 5–30 原子；部分 300 K 驗證改用 2×2×2 超胞；時間步長試行 0.5–1.0 fs；每條探索軌跡 2–10 ps；每隔 0.2–1 ps 抽取若干快照，並統一 0 K 弛豫。MD 中的 A100 等效 GPU 耗時需按模型組合及實測每步耗時換算。

設初期可用 300 個 Diffusion 粗胚、3 種主策略、每策略約 1–2 個重啟、每條平均 5 ps、1 fs 時間步。若以約 1,200 條 MD 軌跡作為擴展實驗的可選規模，則約為：

$$
N_{\mathrm{MD\ steps}}
=1{,}200\times\frac{5\ \mathrm{ps}}{1\ \mathrm{fs}}
=6.0\times 10^6\ \mathrm{steps}.
$$

若實測每個 MD step 的等效 GPU 耗時為 $0.04$–$0.25$ 秒／step，則原始積分約為 67–417 A100 GPU·h，尚未包含 Diffusion、0 K 逐幀弛豫、300 K 超胞 MD、模型微調與排程空閒成本。對連續偏置網路每步反向求導，須量測合併模型而非僅測 MLIP。

### 7.2 規劃資源表

| 計算模組 | 建議規模／假設 | 預估資源（規劃區間） |
| --- | --- | --- |
| Diffusion 生成與少量微調 | 預訓練 checkpoint；約 2,000–5,000 個候選；至多 1–2 輪 adapter 微調 | 80–350 A100 GPU·h |
| MLIP 搜索／退火／快照弛豫 | 依 6 百萬 MD step 規模及實測負載；包括部分重啟與多快照 | 100–650 A100 GPU·h |
| 300 K MD 與超胞測試 | Top 30–60；每個 10–20 ps；不同原子數成本差距較大 | 20–180 A100 GPU·h |
| 預測器重訓、驗證及冗餘 | 小模型 fine-tune、ensemble、重算和失敗重試 | 50–250 A100 GPU·h |
| 合計建議預留 | 優先共享 1–2 張 A100 80GB；可隨時降級 | 約 300–1,500 A100 GPU·h |
| DFT 靜態／弛豫／性質 | Top 50–100；按每個 300–1,500 CPU core·h 粗估 | 約 15,000–150,000 CPU core·h |
| DFT 進階檢查 | 少數聲子／磁態／更高精度計算 | 另預留 5,000–30,000 CPU core·h（視需求） |
| RAM / 儲存 | 管理節點 64–128 GB RAM，稀疏存軌跡與 checkpoint | 建議 0.5–1 TB SSD |

說明：GPU·h 是以 A100 為參考的單 GPU 使用時數，不是同時擁有多 GPU 的日曆時長；若硬體非 A100，須以實測 throughput 換算。DFT core·h 與 GPU·h 不能相加。高溫 MD 是否可信首先取決於 MLIP 的高能／相變區域誤差，單純增加算力無法彌補錯誤勢能面。

### 7.3 最低可行與標準方案

- 最低可行（Pilot-first）：50–100 粗胚；3 組主要策略；2–5 ps 軌跡；少量 Top-20–30 DFT；暫不回訓 diffusion。約 100–350 A100 GPU·h、6,000–40,000 core·h，用以做 Go / No-Go。

- 標準方案：200–300 粗胚，多起點和 5–10 ps 主搜索；Top-50–100 DFT；有條件進行 1–2 輪生成微調。約 300–1,500 A100 GPU·h、20,000–180,000 core·h（含可選進階驗證）；按 W1 profiling 調整。

## 8. 主要風險、應對與備選結果

| 風險 | 可能後果 | 具體控制 |
| --- | --- | --- |
| 過熱、熔化或形成非晶 | 搜索離開目標晶體族 | 自適應溫度、小幅熱脈衝、超胞與局部結構追蹤 |
| MLIP 在高能態軟化／外推 | MD 跨越的是錯誤勢能面 | 抽查代表性的 DFT 力／能量、雙 MLIP 比較、拒絕 OOD 快照 |
| Predictor 在熱態梯度失真 | 偏置 MD 導向代理漏洞 | 無偏置 MD + basin-level ranking 作為主要基線；梯度裁剪和不確定性約束 |
| 0 K 回彈至原盆地 | Property 改善消失 | 強制無偏置弛豫後評估；報告 basin change 與 retention |
| 300 K MD 成本高且時間短 | 檢驗結果無法外推到長期穩定 | 僅做 Top-K 分層檢驗，明確聲稱為 finite-time robustness |
| DFT 篩選偏差 | top-M 不能代表全分佈成功率 | 固定驗證配額、保留少量隨機候選、分別報告 proxy / verified metrics |
| 回流後模式坍縮 | 多樣性／原型覆蓋變差 | 原始資料混合訓練、prototype split、去重與多樣性採樣 |
| 過多研究支線 | 三個月難以收斂 | W3、W7 雙節點決策；不做 Flow Matching，不從零訓練大模型 |

**即使 continuous bias 沒有優勢，本研究仍可形成有價值的負結果：**證明在某些晶體性質下，將 MD 盆地探索與弛豫後屬性評估解耦，會比不可靠的熱態 virtual force 更穩健。只有在獨立驗證顯示正收益時才啟動迭代回訓，是本方案的核心風險控制。

## 9. 預期成果與驗收標準

1. 一套可重現的 Crystal Diffusion → MD/Annealing → 多快照 0 K Relax → 300 K MD／DFT 驗證流程，具標準資料記錄與去重。

2. 一份公平比較 B0–B5 的實驗分析，明確定量物理 basin 覆蓋、property hit retention、Best-of-N 效應與每單位計算成本。

3. 至少一個材料子空間的獨立 DFT 小樣本驗證，並清楚區分穩定性代理、DFT 驗證及短時熱魯棒性。

4. 視 W7 的 Go / No-Go 結果，完成 1–2 輪 Diffusion 回訓消融，或改為提交詳盡的反例及方法邊界研究。

5. 一份方法／實驗論文草稿與可重現設定檔；不將實驗合成、普遍跨體系適用性或高品質新材料發現列為三個月必達成果。

## 參考文獻

- [1] Xie, T. et al. Crystal Diffusion Variational Autoencoder for Periodic Material Generation. ICLR (2022). <https://arxiv.org/abs/2110.06197>

- [2] Jiao, R. et al. Crystal Structure Prediction by Joint Equivariant Diffusion. NeurIPS (2023). <https://proceedings.neurips.cc/paper_files/paper/2023/hash/38b787fc530d0b31825827e2cc306656-Abstract.html>

- [3] Zeni, C. et al. A generative model for inorganic materials design. Nature 639, 624–632 (2025). <https://doi.org/10.1038/s41586-025-08628-5>

- [4] Amsler, M. & Goedecker, S. Crystal structure prediction using the minima hopping method. J. Chem. Phys. 133, 224104 (2010). <https://doi.org/10.1063/1.3512900>

- [5] Martoňák, R., Laio, A. & Parrinello, M. Predicting Crystal Structures: The Parrinello-Rahman Method Revisited. Phys. Rev. Lett. 90, 075503 (2003). <https://doi.org/10.1103/PhysRevLett.90.075503>

- [6] Accelerated identification of equilibrium structures of multicomponent inorganic crystals using machine learning potentials. npj Comput. Mater. (2022). <https://www.nature.com/articles/s41524-022-00792-w>

- [7] Deng, B. et al. CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling. Nat. Mach. Intell. 5, 1031–1041 (2023). <https://doi.org/10.1038/s42256-023-00716-3>

- [8] Fujii, A. et al. Adaptive Constraint Integration for Simultaneously Optimizing Crystal Structures with Multiple Targeted Properties (SMOACS). arXiv:2410.08562 (2024; versions updated). <https://arxiv.org/abs/2410.08562>

- [9] Han, X.-Q. et al. InvDesFlow-AL: active learning-based workflow for inverse design of functional materials. npj Comput. Mater. 11, 364 (2025). <https://doi.org/10.1038/s41524-025-01830-z>

- [10] Deng, B. et al. Systematic softening in universal machine learning interatomic potentials. npj Comput. Mater. 11, 9 (2025). <https://doi.org/10.1038/s41524-024-01500-6>
