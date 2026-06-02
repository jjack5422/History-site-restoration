# Run: synth-v0

- 日期: 2026-06-02
- 目的: v0 正式合成 crack:craq = 8:1 (crack 800 / craq 100 fg tiles)
- 指令: `python scripts/synthesize.py --config configs/synth_v0.yaml --overwrite`
- config: configs/synth_v0.yaml(seed 42)
- 環境: venv /home/zzz90/research/datagen_env;crackseg_common editable
- 輸出: _data/synth_crack_v0, _data/synth_craq_v0(TileSegDataset 格式)
- 驗收: TileSegDataset 可讀=是;fg 分佈 within real range crack=true/craq=true;視覺=通過
- 結果數字: crack tiles=806 reuse=3;craq tiles=108 reuse=0
- 備註: 下游 real vs real+synth A/B 為後續獨立實驗(本 run 不含)
