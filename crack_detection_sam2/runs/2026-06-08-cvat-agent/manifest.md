# 2026-06-08 CVAT craq+crack assisted-labeling AI agent

## 目的
把既有 final experts 包成 app.cvat.ai 的本機輔助標註工具,只產 **craquelure** + **crack** 兩類 mask,
讓使用者改模型輸出而非從零畫。模型在本機跑,只有 function metadata 上傳 CVAT。

- Spec: `docs/superpowers/specs/2026-06-08-cvat-craq-crack-agent-design.md`
- Plan: `docs/superpowers/plans/2026-06-08-cvat-craq-crack-agent.md`
- 程式: `crack_detection_sam2/cvat_agent/`(`craq_crack_func.py` + `_env_smoke.py` + 測試 + register/run 腳本 + README)

## 環境(venv `cvat_agent_env`, Python 3.12.3)
```bash
cd /home/zzz90/research
python3 -m venv cvat_agent_env
cvat_agent_env/bin/pip install --upgrade pip
cvat_agent_env/bin/pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
cvat_agent_env/bin/pip install -r crack_detection_sam2/cvat_agent/requirements.txt   # 含 albumentations(crackseg_common.augment 間接相依)
cvat_agent_env/bin/pip install -e /home/zzz90/research/segment-anything-2 --no-build-isolation
cvat_agent_env/bin/pip install -e /home/zzz90/research/_lib/crackseg_common
cvat_agent_env/bin/pip install pytest   # 測試用
```
關鍵版本: torch 2.11.0+cu128 / torchvision 0.26.0+cu128 / smp 0.5.0 / cvat-sdk 2.67.0 / cvat-cli 2.67.0。
模型: craq = `runs/expert_craq_v3_final_small/last.pt`(SAM2 Hiera-small dense-seg);
crack = `crack_detection_unet/runs/expert_crack_v3_final_resnet50/last.pt`(ResUNet-50)。

## 測試結果(對照 spec §7)
1. **Env smoke(離線)**: `cvat_agent_env/bin/python crack_detection_sam2/cvat_agent/_env_smoke.py`
   → `ENV SMOKE OK ... craq (2,1024,1024) crack (2,1024,1024)`。**符合** — smp 與 SAM-2 共存、8GB GPU 容得下兩模型。
2. **Function smoke(離線)**: `pytest test_craq_crack_func.py` → `1 passed`。回傳合法 CVAT mask shapes,
   `encode_mask`→`decode_mask` 還原為原圖尺寸非空遮罩。**符合**。
   (修正: 測試斷言 `s.type == "mask"` 改 `str(s.type) == "mask"`,因 cvat_sdk `.type` 為 `ShapeType` enum。)
3. **Live smoke(CVAT)**: **符合**。
   - app.cvat.ai 免費方案的「背景/agent automatic annotation」被鎖(需付費),故改走 `cvat-cli task auto-annotate`
     (本機跑 function、經一般 REST API 寫回 task),不需付費功能。
   - 註冊: `function create-native "Heritage craq+crack"` → **function-id #4747**(metadata 已上傳;此路徑因方案限制未實際用 agent 跑)。
   - 實跑指令:
     ```bash
     export CVAT_ACCESS_TOKEN=<PAT>
     cvat_agent_env/bin/cvat-cli --server-host https://app.cvat.ai \
       task auto-annotate 2282525 --function-file crack_detection_sam2/cvat_agent/craq_crack_func.py
     ```
   - 對象 task **2282525**(`annotation_batch_2`, 8 frames),**附加模式**(未加 `--clear-existing`)。
     8/8 影格成功,`Upload complete`。task **2240171 未碰**。
   - 寫入統計(查 task annotations): crack mask **731**、craquelure mask **585**、(原有)background 8 + shrinkage 8。
     8 張影格皆有兩類 mask 供檢視。

## 結論
端到端可用:離線兩項 smoke + 線上 auto-annotate 全部**符合** spec §7 預期。craquelure 與 crack mask
能寫回 CVAT task 供人工檢視/修改。

## 已知限制 / 待查
- 免費方案無法用 UI 的背景 automatic annotation(`run-agent` 路徑);改用 `task auto-annotate` 達成同效。
  若日後升級付費,可直接 `register.sh` + `run_agent.sh` 走 agent 路徑。
- crack 偏破碎(8 張切出 731 個連通元件),品質弱於 craquelure(spec §8 已記)。降噪可調
  `-p min_blob=int:256 -p thresh_crack=float:0.6`。
- PAT 已在對話中明文出現,使用者應於 app.cvat.ai 撤銷/重新產生。
