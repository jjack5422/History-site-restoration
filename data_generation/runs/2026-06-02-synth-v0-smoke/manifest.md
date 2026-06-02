# Run: synth-v0-smoke

- 日期: 2026-06-02
- 目的: 端到端煙霧驗證合成 pipeline(小規模 crack 16 / craq 2)
- 指令: `python scripts/synthesize.py --config configs/synth_smoke.yaml --overwrite`
- config: configs/synth_smoke.yaml(seed 42, 8:1)
- 環境: venv /home/zzz90/research/datagen_env;crackseg_common editable
- 前置: data/base_manifest.json, data/appearance/appearance_profile_{crack,craq}.json
- 輸出: _data/synth_{crack,craq}_smoke(煙霧用,可刪)
- 結果: crack tiles=18 (fg=18) samples=2 reuse=0; craq tiles=9 (fg=9) samples=1 reuse=0;TileSegDataset 可讀 = 是;視覺抽檢 = 通過(crack fg%=0.62, craq fg%=8.41)
