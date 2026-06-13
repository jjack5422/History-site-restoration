# 龜裂半自動標註管線設計（草稿→人工修正）

日期：2026-06-14
範圍：30 張古蹟彩繪原圖（`_data/image/*.jpg`），本輪只標 **龜裂 craquelure**（裂縫 crack 下一輪）。

## 問題

- 原圖超高解析度（6,000–16,000 px 寬），龜裂網線寬僅 1–3 px，部分 < 1px。
- 龜裂常與彩繪顏料、線條、髒污疊在一起，難分（內容型 FP，見 memory `finding_craq_refiner_content_fp`）。
- 標註面積大、單人作業，耗時。瓶頸在「產生 GT 標註」而非推理。
- 現有 `tile→影像處理粗抓→CVAT 精修` 流程，精修時間一半補細網、一半刪錯報。

## 核心想法

用 **人畫的粗「龜裂區域」多邊形** 同時做兩件事：
1. **區域外一刀刪 FP**：邊框/浮雕/題字/空白落在框外，自動清除。
2. **區域內把 recall 拉到平時全圖不敢用的激進檔**：因 FP 已被框死，區域內可低門檻猛抓，補滿細網，人幾乎不用逐條描。

把「全圖逐條精修」換成「畫幾個粗區域圈 + 收尾」。

## 兩種區域（這 30 張約一半一半）

- **乾淨龜裂區**（素色底/背景）：草稿用多尺度 ridge filter（Sato/Meijering，dark line）開高 recall。框外無內容可抓錯。
- **疊圖龜裂區**（龜裂長在人物/題字/花葉上）：草稿改用訓練好的 ResUNet 輸出（學過裂縫長相，壓掉部分筆畫），疊信心熱力圖，人只判模型「猶豫的那一條帶」。

## 管線

1. **切圖**：原解析度切 512×512、重疊 64，沿用現有 tile 流程。
2. **草稿生成器（stage-0，本輪新增 `crack_detection_sam2/prelabel_draft.py`）**：
   每 tile / 每圖 = `模型通道 ∪ ridge 通道`：
   - 模型：ResUNet craq prob（`crack_detection_unet` `predict_full --save_prob`，channel 1），低門檻 ~0.15 求 recall；可再接 SAM2 refine。
   - ridge：多尺度 Sato（σ≈1–3px），對暗細線高 recall 低 precision。
   - 後處理：去 < min_area 碎點、小閉運算連碎片、可選 skeletonize 細化貼回 1px。
   - 宁滥勿缺，FP 交給區域裁。
3. **CVAT 修正**：逐 tile 上傳兩層 — `craquelure`（草稿 mask）+ 人畫 `craq_region`（區域多邊形，分乾淨/疊圖兩種）。
4. **合併匯出**：`mask ∩ region` → 拼回原圖座標成 GT。
5. **Active learning（分輪）**：先標龜裂最重幾張（尤其疊圖區）→ 微調 craq 模型（用上既有 DINOv2 特徵，壓內容 FP）→ 重出剩餘草稿（更準）→ 再修；可選再回爐一次。

## 明確不做（YAGNI）

- VLM 畫 mask / VLM 框區域：對「補細網」痛點無用，退場。
- crack 中心線折線：線狀 crack 才用，龜裂網狀不用；crack 本輪不做。

## 老實的時間預期（非魔法）

- 刪錯報半 → 畫粗區域，砍掉約 80%+。
- 補細網半 → 靠「區域內激進 recall + 只判猶豫帶」再砍一部分，前提是接受「區域內草稿夠好就收」，不逐條像素級完美。堅持完美則只能靠 active learning 慢慢餵準模型。

## 第一版 prelabel 觀察（2026-06-14，KJTHT-SC-L-1RB1-1 花卉）

- 乾淨區：模型描出明顯網裂主線，但線偏粗（2-4px）、細網漏不少；ridge@pct96 補得保守。
- 疊圖區：亮底網裂尚可；暗葉內部網裂幾乎漏；ridge 未大量爆在葉上。
- 待調：降 ridge 門檻補細網（區域裁剪使其安全）、skeletonize 細化、邊框/浮雕 FP 由區域圈清除。
