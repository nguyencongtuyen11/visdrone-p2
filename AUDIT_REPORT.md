# BAO CAO AUDIT RL-SAHI — 21 van de xac nhan (sau dedup)

## [CRITICAL] benchmark_speed.py:119
**T4 speed benchmark breaks RL loop on the FIRST rejected rollout: _attempt_overlap is computed AFTER appending roi, so it always returns 1.0**

- Van de: In the inline RL-HYBRID loop, when a rollout is rejected (max_steps/stalled/overlap), the roi is appended to `att` BEFORE calling `_attempt_overlap(roi, att)`. `_attempt_overlap` (pipeline.py:153-164) computes max intersection/area(roi) against the list — which now contains roi itself, so intersection == own area and the result is exactly 1.0 >= 0.95, triggering `break` unconditionally. The produc
- Phan bien xac nhan: Confirmed by reading the code. In Test/scripts/benchmark_speed.py:117-119 the rejected roi is appended to `att` BEFORE calling `_attempt_overlap(roi, att)`; since `_attempt_overlap` (pipeline.py:153-164) divides the max intersection by area(roi) and the list now contains roi itself, the result is exactly 1.0 >= 0.95, so `break` fires on the FIRST rejected rollout and the `continue` on line 120 is 
- **FIX:** Mirror production ordering: compute `ov = _attempt_overlap(roi, att)` first, then `att.append(roi)`, then `if ov >= 0.95: break` else `continue`. Re-run the T4 benchmark afterwards, since published RL-HYBRID latency/recall numbers were produced by the truncated loop.

## [CRITICAL] benchmark_speed.py:119
**RL-HYBRID break ngay lần reject đầu tiên: _attempt_overlap tính overlap của roi với list chứa CHÍNH NÓ**

- Van de: Trong vòng RL-HYBRID, khi rollout bị reject thì code append roi vào att TRƯỚC rồi mới gọi _attempt_overlap(roi, att). Vì att lúc này chứa chính roi, intersection(roi, roi) = area(roi) nên hàm luôn trả ~1.0 >= 0.95 và break — vòng thử lát RL kết thúc ngay ở lần reject ĐẦU TIÊN thay vì tiếp tục tới max_attempts=14 như production.
- Phan bien xac nhan: Bug thật, đã đối chiếu source. `_attempt_overlap` (pipeline.py:153-164) tính inter.max()/area(roi) qua intersection_matrix (box_geometry.py:22-39) — KHÔNG loại self. benchmark_speed.py:118-119 append `roi` vào `att` TRƯỚC rồi gọi `_attempt_overlap(roi, att)`, nên att chứa chính roi → self-intersection = area(roi) → tỷ lệ luôn = 1.0 ≥ 0.95 → break ngay ở lần reject ĐẦU TIÊN thay vì chạy tới max_att
- **FIX:** Sửa dòng 119 thành `if _attempt_overlap(roi, att[:-1]) >= 0.95: break` (loại self, khớp benchmark_oneshot.py), hoặc tính overlap TRƯỚC khi append như production benchmark.py. Sau khi vá phải đo lại toàn bộ số RL-HYBRID (latency/recall/crops) vì các số cũ đều lệch.

## [MAJOR] slice_env.py:819
**Dead knobs: tinh chinh round-1 recall-max cho step_penalty va area_penalty trong rl.yaml la NO-OP vi simplified reward khong doc chung**

- Van de: rl.yaml dat use_simplified_reward: true, nhung _simplified_reward hardcode chi phi buoc `step_cost = 0.05 + roi_area_ratio * 0.5` nhan voi efficiency_weight (slice_env.py:819-820); `step_penalty` chi duoc doc trong _legacy_reward (dong 887) va `area_penalty` cung chi o legacy (dong 930). Round-1 da doi step_penalty 0.03->0.01 voi comment 'buoc di re hon -> dam kham pha lau hon' va area_penalty 0.3
- Phan bien xac nhan: Confirmed. rl.yaml:40 bật use_simplified_reward và _reward (slice_env.py:754) đi vào _simplified_reward, nơi step cost bị hardcode `0.05 + roi_area_ratio * 0.5` nhân efficiency_weight (slice_env.py:819-820). Grep toàn bộ Test/src cho thấy step_penalty chỉ được đọc ở slice_env.py:887 và area_penalty chỉ ở slice_env.py:930 — cả hai nằm trong _legacy_reward không bao giờ chạy. Vậy hai tinh chỉnh roun
- **FIX:** Cho _simplified_reward đọc config thay vì hardcode: `step_cost = cfg.step_penalty + roi_area_ratio * cfg.area_penalty` (giữ efficiency_weight làm hệ số chung), đồng thời chú thích trong rl.yaml các knob legacy-only hoặc xóa chúng khi use_simplified_reward=true. Bỏ (hoặc tính lười sau cờ debug) khối compactness_delta ở slice_env.py:773-785 vì chỉ gh

## [MAJOR] trainer.py:178
**Terminal reward zeroes all positive crop outcome (including tp_gain) for accepted crops that miss hard boxes, punishing slices that genuinely add new TPs**

- Van de: When a crop is accepted by the utility rule but hits no GT hard box, the reward is `min(base_reward, 0.0) + negative_crop_reward - accepted_no_hard_penalty`, where `negative_crop_reward = min(outcome.reward, 0.0) * scale` (line 171). outcome.reward contains `tp_reward*tp_gain + detection_reward*utility` (crop_outcome.py:214-216), but its positive part is discarded entirely. Hard boxes are only GT 
- Phan bien xac nhan: Confirmed by reading the code. In trainer.py:177-178, an ACCEPTED crop with hard_new_hits==0 gets `min(base_reward,0) + min(outcome.reward,0)*0.1 - 2.0`, which is always <= -2.0: the min(...,0) wipes out the only GT-grounded positive signal (`tp_reward*tp_gain`, crop_outcome.py:214-216, matched against real labels via IoU in _tp_fp_gain/_match_counts). Meanwhile that same accepted crop's boxes ARE
- **FIX:** In the `hard_new_hits == 0 and outcome.accepted` branch, stop discarding the GT-verified positive signal: use the full signed crop reward, e.g. `return float(min(base_reward, 0.0) + crop_reward - cfg.accepted_no_hard_penalty)`, or at minimum add back `crop_scale * evaluator.tp_reward * outcome.tp_gain` and only apply accepted_no_hard_penalty when `

## [MAJOR] merge.py:207
**Boundary-truncated crop boxes are never filtered: counted as 'novel' gains and survive final NMS as duplicate FPs alongside the full-image box of the same object**

- Van de: run_yolo_on_crops (crops.py:109-133) keeps every detection in the crop, including objects cut by the ROI edge (crop_roi truncates at ROI bounds, crops.py:24-31 — there is no filter for boxes touching the crop border, unlike standard SAHI). A truncated partial box of a large object typically has IoU < 0.5 with the complete full-image box, so: (a) it survives class_aware_nms at merge_iou=0.5 (merge.
- Phan bien xac nhan: Confirmed by reading the code. crop_roi (crops.py:24-32) hard-truncates at ROI bounds and run_yolo_on_crops keeps every detection — no border filter, no center_inside/IoA use anywhere in the inference path. Both nms_numpy and the duplicate check (merge.py:212-215) use plain IoU; a partial box covering <~50% of a large object has IoU<0.5 with the full-image box, so it survives class_aware_nms (merg
- **FIX:** In _novel_candidate_detections_after_merge, mark a candidate as duplicate when max(IoU, IoA) >= threshold using the existing ioa_matrix (inter / area of the candidate box), so contained partials (IoA≈1) are never counted novel. Additionally (or alternatively), in run_yolo_on_crops drop detections whose boxes touch a crop border that is not also an 

## [MAJOR] benchmark.py:615
**benchmark_split latency excludes the full-image YOLO pass for every method (yolo_full latency is just numpy masking), and rl_sahi crops_per_image omits rejected crops that still ran YOLO**

- Van de: In benchmark_split, `det = get_initial_detection(...)` (lines 598-613) runs OUTSIDE all timers; the yolo_full timer (lines 615-617) wraps only `_full_predictions(det, infer_cfg)` — a boolean mask over cached arrays (~0 ms). With use_cache=True (config default) the full-image YOLO forward never even executes. The SAHI/topk/rl timers likewise exclude the initial detection they depend on. Separately,
- Phan bien xac nhan: Confirmed by reading the code. (1) benchmark.py:598-613 computes det=get_initial_detection() outside every timer; the yolo_full timer (615-617) wraps only _full_predictions(), which is pure numpy masking (lines 117-121). With use_cache=True (scripts/benchmark.py:117 default, configs/inference.yaml:26) the full-image YOLO forward is loaded from disk cache and never timed, so yolo_full latency repor
- **FIX:** Move the timer start above get_initial_detection (or time the base detection once per image with use_cache=False and add that duration to every method's latency), so each method's latency includes the full-image pass. In _predict_rl_sahi, increment an executed_crops counter at each run_yolo_on_crop call and return it (instead of len(accepted_rois))

## [MAJOR] benchmark_hybrid.py:94
**RL-mode hybrid benchmark silently uses dataclass defaults (min_slice_utility=0.5, max_slice_attempts=0) instead of production inference.yaml values**

- Van de: The InferenceConfig built at benchmark_hybrid.py:94-100 passes only imgsz/conf/iou/merge/max_det/device/feature_layers/target_classes/class_mapping. It omits min_slice_detections, min_slice_utility, duplicate_iou, max_slice_attempts, require_stop_for_acceptance, so `_predict_rl_sahi` runs with InferenceConfig defaults (config.py:23-28): min_slice_utility=0.5 (production inference.yaml uses 0.2), m
- Phan bien xac nhan: Confirmed. benchmark_hybrid.py:94-100 omits the five acceptance knobs when building InferenceConfig, so in --fine-mode rl, _predict_rl_sahi (eval/benchmark.py:269, 351-355) runs with dataclass defaults: min_slice_utility=0.5 instead of the tuned 0.2 from inference.yaml (which IS loaded via default.yaml includes but never read), and max_slice_attempts=0 -> 2*max_slices=16 attempts instead of yaml's
- **FIX:** In benchmark_hybrid.py:94-100, mirror scripts/benchmark.py:95-100: add min_slice_detections=int(ic.get("min_slice_detections", 1)), min_slice_utility=float(ic.get("min_slice_utility", 0.5)), duplicate_iou=float(ic.get("duplicate_iou", ic.get("merge_iou", 0.5))), max_slice_attempts=int(ic.get("max_slice_attempts", 0)), require_stop_for_acceptance=bo

## [MAJOR] batched_trainer.py:577
**Reward exploit: farm hard_hit_reward 4.0 nhieu lan tren CUNG mot hard-region qua cac attempt bi reject, vi covered chi duoc luu khi slice duoc chap nhan**

- Van de: Khi slice bi reject (crop khong them detection), `w.previous_covered` KHONG duoc cap nhat (batched_trainer.py:577-579 chi cap nhat trong nhanh `if not reject_slice`), nhung `_terminal_reward_with_crop_outcome` van tra FULL `hard_hit_reward * hard_new_hits` cho attempt bi reject do (trainer.py:173-176, chi tru rejected_crop_penalty 0.5). Attempt ke tiep tao env moi voi previous_covered cu (batched_
- Phan bien xac nhan: Xác nhận sau khi đọc code thật. Chuỗi exploit tồn tại đầy đủ: (1) batched_trainer.py:577-579 chỉ cập nhật w.previous_covered khi slice được chấp nhận; (2) dòng 503-509 vẫn tính hard_new_hits so với previous_covered cũ và trainer.py:173-176 trả full hard_hit_reward*hits (4.0) cho attempt bị reject, chỉ trừ rejected_crop_penalty 0.5 (crop reward âm chỉ ~-0.12 do scale 0.1); (3) env attempt mới (dòng
- **FIX:** Thêm mask w.rewarded_covered vào EnvWorker, cập nhật nó sau MỌI attempt done (kể cả reject): rewarded_covered |= env.covered; và tính terminal_hard_new_hits = (env.covered & ~(previous_covered | rewarded_covered)) tại batched_trainer.py:503 (tương tự new_hits dòng 558 nếu muốn nhất quán). Cách tối giản hơn: chỉ trả hard_hit_reward khi outcome.accep

## [MAJOR] batched_trainer.py:273
**Resume khong kiem tra env_cfg khop voi checkpoint: max_steps da drift (checkpoint 10 vs rl.yaml 20) lam lech chuan hoa state neu resume**

- Van de: Duong resume (batched_trainer.py:273-300) validate state_dim (dong 275) va so action (dong 280-285) nhung KHONG validate env_cfg; env_cfg luon lay tu rl.yaml hien tai (scripts/train.py:58 `cfg.dataclass_instance("env", EnvConfig)`). Bang chung drift da xay ra: best.pt luu env_cfg max_steps=10 trong khi configs/rl.yaml:21 hien la `max_steps: 20` (va EnvConfig default la 20/8 — env_config.py:12-13).
- Phan bien xac nhan: CONFIRMED with one factual correction. The core defect is real: batched_trainer.py:273-300 validates only state_dim and action count on resume; env_cfg/state_cfg are never compared even though _save_resume_checkpoint writes env_cfg into resume.pt (line 125), env_cfg always comes from the current rl.yaml (scripts/train.py:58), and the replay buffer is restored wholesale (line 291). The steps_norm m
- **FIX:** In the resume block (batched_trainer.py:~285), compare resume_data["env_cfg"] and resume_data["state_cfg"] against asdict(env_cfg)/asdict(state_cfg) and raise a RuntimeError listing the differing keys (same "Delete resume.pt or run with --no-resume" guidance as the existing checks); optionally allow an explicit override flag that flushes the replay

## [MAJOR] benchmark.py:43
**Benchmark evaluates only 6 of 10 VisDrone classes — absolute metrics not comparable to the 10-class VisDrone protocol**

- Van de: BenchmarkConfig defaults target_classes=(0, 2, 3, 5, 8, 9) (line 43) and configs/inference.yaml pins the same six (pedestrian, bicycle, car, truck, bus, motor), dropping people(1), van(4), tricycle(6), awning-tricycle(7). Both GT (lines 64-66) and predictions are filtered symmetrically, so the eval is internally consistent — this is intentional config, not a coding bug — but the detector is a 10-c
- Phan bien xac nhan: Confirmed by code reading. benchmark.py line 118 masks full-image detections at cfg.output_conf, and all three crop paths (lines 159, 239, 315) pass conf=cfg.output_conf to run_yolo_on_crop, so every prediction entering _evaluate_method is pre-thresholded at 0.05-0.40 depending on config (0.10 default). _ap_from_pr (lines 371-381) pads recall to 1.0 with precision 0.0, so the recall region beyond 
- **FIX:** Add a separate AP threshold (e.g. map_conf ~0.001-0.01) used only for mAP: for yolo_full reuse the cached full_conf=0.01 detections directly, and run crops with conf=map_conf while keeping output_conf solely for operating-point metrics (recall, fp_per_image). In the thesis, label current mAP50 as "mAP at operating point conf=0.10" and never compare

## [MAJOR] benchmark.py:368
**crops_per_image for RL-SAHI counts only ACCEPTED slices, but YOLO actually runs on every attempted slice — crop-budget comparison with baselines is skewed**

- Van de: _predict_rl_sahi runs run_yolo_on_crop on each candidate ROI (line 310) BEFORE the gain/utility gate (lines 351-355); rejected candidates `continue` after having consumed a full crop inference. Yet the function returns `len(accepted_rois)` (line 368) as crop_count, which benchmark_split records as crops_per_image (line 638). With min_slice_utility=0.2 and max_slice_attempts=24, RL-SAHI can run up 
- Phan bien xac nhan: Confirmed by reading the code. In _predict_rl_sahi (Test/src/rl_sahi/eval/benchmark.py), run_yolo_on_crop executes at line 310 BEFORE the gain/utility gate at lines 351-355; rejected candidates `continue` after the YOLO inference is already spent, yet line 368 returns len(accepted_rois), which benchmark_split records as crops_per_image (lines 638, 653). Baselines report their true inference counts
- **FIX:** In _predict_rl_sahi, add a counter (e.g. yolo_crop_calls) incremented immediately after the run_yolo_on_crop call at line 310 and return both it and len(accepted_rois); record crops_per_image from the inference counter (and optionally accepted_crops_per_image separately), then re-tune benchmark.topk_slices to match the attempted-inference average s

## [MAJOR] trainer.py:321
**best.pt is selected by a GT-assisted geometric coverage proxy whose slice-acceptance rule differs from both training and inference; the intended benchmark selection (small_recall weight 4.0) never ran in the shipped checkpoint**

- Van de: _greedy_eval_episode accepts/rejects slices using GT hard-region coverage (`new_hits < env_cfg.min_new_hits_to_accept`, where new_hits derives from env.covered computed against GT hard_boxes) and stops early on `previous_covered.all()` (line 330). Training instead rejects via `terminal_outcome.accepted` (trainer.py:800, detector-utility based), and inference (inference/pipeline.py:562-563) accepts
- Phan bien xac nhan: Confirmed for the configs actually used to train the reported cloud models. evaluate_policy (trainer.py:346-351, batched_trainer.py:629) draws a fresh with-replacement subset via dataset.random_episode -> random.choice(self.samples) (dataset.py:90) every eval, with no per-eval seed reset (seed set once at trainer.py:481). best.pt is chosen by `selected_score > best_score` (trainer.py:893 / batched
- **FIX:** Evaluate on a fixed deterministic subset (or the whole val split): add a CachedEpisodeDataset.iter_episodes() and have evaluate_policy loop over a stable, seed-frozen list instead of random.choice, so scores are comparable across evals. Additionally, use a dedicated local RNG (random.Random(cfg.seed)) for eval sampling so evaluation does not pertur

## [MAJOR] benchmark_hybrid.py:94
**Chế độ --fine-mode rl chạy _predict_rl_sahi với gate mặc định (utility 0.5, attempts 2*fine_k) thay vì config đã tune (0.2, 24) — dead knob**

- Van de: InferenceConfig ở benchmark_hybrid.py:94-100 chỉ truyền full_imgsz/slice_imgsz/full_conf/output_conf/iou/merge_iou/max_det/device/feature_layers/target_classes/class_mapping — KHÔNG truyền min_slice_utility, min_slice_detections, duplicate_iou, max_slice_attempts, require_stop_for_acceptance. Khi --fine-mode rl gọi _predict_rl_sahi (line 135), gate chạy bằng default của dataclass: min_slice_utilit
- Phan bien xac nhan: Đọc code xác nhận finding ĐÚNG, không bác bỏ được. benchmark_hybrid.py:94-100 dựng InferenceConfig KHÔNG truyền min_slice_utility và max_slice_attempts, nên khi --fine-mode rl gọi _predict_rl_sahi (benchmark.py) thì gate tại dòng 351-354 dùng cfg.min_slice_utility=0.5 (default dataclass) và dòng 269 tính max_attempts = env_cfg.max_slices*2 = fine_k*2 = 16 (vì cfg.max_slice_attempts=0). Trong khi d
- **FIX:** Trong benchmark_hybrid.py khi dựng InferenceConfig, đọc và truyền các tham số gate từ section infer đã load (giống benchmark_speed/oneshot): thêm min_slice_utility=float(ic.get("min_slice_utility",0.5)), min_slice_detections, duplicate_iou, max_slice_attempts=int(ic.get("max_slice_attempts",0)), require_stop_for_acceptance. Như vậy chế độ --fine-mo

## [MINOR] trainer.py:176
**hard_hit_reward is farmable: rejected slices still earn full hard-hit + base reward, and rejection never commits coverage, so the same GT box is re-credited across attempts**

- Van de: In _terminal_reward_with_crop_outcome, a slice with hard_new_hits > 0 gets the full base reward (which already contains target_reward*new_hits + density + stop bonus) plus hard_hit_reward*hits (4.0 each per rl.yaml) even when the crop outcome is REJECTED (utility < min_slice_utility) — the only difference vs an accepted slice is rejected_crop_penalty = 0.5. Then in the attempt loop, the rejected p
- Phan bien xac nhan: Fact confirmed: state_summary.py:82 clips summary[23]=min(attempted_count/10,1.0) while inference runs up to 24 attempts (inference.yaml max_slice_attempts=24, pipeline.py:385) and needs >=14 attempts to fill max_slices=14, so the channel pins at 1.0 from attempt 10 onward in normal operation, and no other feature encodes remaining budget. However the claimed impact is overstated: (1) each attempt
- **FIX:** Derive slice_count_norm from the actual attempts cap (e.g. set to max_slice_attempts=24, or pass (max_attempts - attempts)/max_attempts as a remaining-budget feature) and retrain — but expect negligible gain unless the episode formulation is also changed to span all attempts of an image so budget can enter the return.

## [MINOR] benchmark.py:567
**Small-object threshold is a percentile of the LIMITED image subset — 'small' definition changes with --limit, breaking run-to-run comparability**

- Van de: benchmark_split first applies limit (`images = iter_images(image_root, split=split, limit=limit)`, line 558) and then computes `small_threshold = _small_area_threshold(images, ...)` (lines 567-574) as the 40th percentile of GT area ratios over ONLY those images. A test-100 run and a full-split run therefore use different area cutoffs, so their small_recall values are not measuring the same populat
- Phan bien xac nhan: Confirmed. Ultralytics 8.4.6 Detect head (head.py:139) returns dict(scores=..., feats=...) computed on the letterboxed input (LetterBox auto=True, center padding to stride multiple — augment.py:1606-1624), and _resize_tensor_maps (features.py:124-136) interpolates the whole map including padding to 16x16 without cropping. Meanwhile detection/history/ROI/slice channels rasterize original-image coor
- **FIX:** In _extract_detect_aux, compute the letterbox content region from image_shape + imgsz + stride (same formula as ultralytics LetterBox) and crop each feature level (and the reshaped score map) to that region before calling _resize_tensor_maps — e.g. F.interpolate on feature[..., top_cells:h-bottom_cells, left_cells:w-right_cells] (or grid_sample for

## [MINOR] ft_rl.yaml:20
**Cac trong so chon best.pt kieu recall-max trong rl.yaml (eval_small_recall_weight=4.0...) la DEAD KNOB trong moi config train thuc te — best.pt duoc chon bang coverage hinh hoc, khong phai small-recall**

- Van de: rl.yaml:92-98 tinh chinh ky luong cach chon best.pt theo huong recall-max (eval_map_weight=0.5, eval_small_recall_weight=4.0 voi ghi chu '★ chọn best.pt CHỦ YẾU theo small-recall', eval_fp_cost_weight=0.0). Nhung benchmark_score (dung cac trong so nay) chi chay khi eval_benchmark_images > 0 (src/rl_sahi/rl/batched_trainer.py:206 va 634-656). Trong khi do TAT CA config train dang dung deu dat eval_
- Phan bien xac nhan: CONFIRMED. count_norm=100 (state_config.py:23) vs cache build conf=0.01/max_det=3000 (cache_builder.py, ft.yaml) — kiem tra lai tren cache that (Test/data/cache_ft/detections) tai tao dung so cua reviewer: train mean 236.6 dets/anh (77.1% >=100), test 298.1 (87.5% >=100), proposals 209.6 (72.1% >=100). Khong co gi cuu: target_classes=[0..9] + class mapping dong nhat nen _filtered_detections khong 
- **FIX:** Nang count_norm len ~500-1000 hoac dung log-scale (vd log1p(count)/log1p(count_norm)) cho summary[0,3,4,24] (va can nhac roi_count_norm tuong tu); thay doi nay chi co tac dung sau khi train lai policy.

## [MINOR] benchmark.py:436
**small_recall has no one-to-one matching: one prediction can validate multiple GT boxes and matched predictions are reused**

- Van de: The small-recall loop (lines 427-439) counts a small GT as hit if ANY same-class prediction has IoU >= 0.5 with it, using `iou_matrix(gt_box.reshape(1,4), boxes[pred_mask]).max()`. There is no assignment: the same prediction box can be counted as the match for several overlapping GT boxes, and predictions already 'used' by other GTs (small or large) are not excluded. The `matched_by_image` bookkee
- Phan bien xac nhan: Xac nhan ve mat code: vong small_recall (dong 427-439) khong lam matching 1-1. No dem moi small GT la "hit" neu BAT KY prediction cung class co IoU>=0.5, khong duy tri tap prediction da dung; matched_by_image cua vong AP la per-class va khong tai su dung o day. Vi vay small_recall thuc chat la "ty le duoc phu" (coverage), la can tren cua recall 1-1 thuc su, va co the bi thoi phong khi 1 prediction
- **FIX:** Doi small_recall sang greedy 1-1 giong vong AP: trong moi anh, chi xet cac prediction cung class, sort theo score giam dan, gan moi prediction cho GT nho co IoU>=0.5 cao nhat va chua bi gan, dung mot mask "used" de moi prediction chi tinh 1 lan. Hoac neu chu y la coverage thi doi ten metric thanh "small_coverage" va ghi ro dinh nghia trong bao cao 

## [MINOR] pipeline.py:467
**No fallback when the policy never selects STOP: up to 24 full rollouts are burned and the image silently degrades to full-image-only output**

- Van de: With require_stop_for_acceptance=true, a rollout ending via max_steps (info from slice_env.py:151-155) is rejected before any crop is run (pipeline.py:467-489, `continue`). If the policy fails to STOP on an image, the loop repeats up to max_slice_attempts=24 rollouts, each costing max_steps+1=11 forward passes plus valid_actions (10 candidate-ROI overlap evaluations per step, slice_env.py:165-185)
- Phan bien xac nhan: Doi chieu code: co so cua finding la THAT, khong bac bo duoc phan co che. (1) pipeline.py:467-489 dung nhu mo ta: khi require_stop_for_acceptance=true (da xac nhan trong inference.yaml:15) va info["stop_due_to_max_steps"]=True (slice_env.py:151-155: agent het max_steps ma chua bao gio chon STOP), ROI bi day vao rejected_rois va `continue`, KHONG chay crop. (2) Vong lap co the lap den max_slice_att
- **FIX:** Them circuit-breaker: dem so lan tu choi lien tiep vi non-STOP (max_steps/stalled) va break outer-loop sau ~2-3 lan de cat lang phi rollout. Va/hoac them fallback chat luong khi accepted_rois rong: hoac accept ROI bi tu choi tot nhat, hoac goi topk-objectness slicing (da co san trong eval/benchmark.py:182-209) de anh khong bi thut ve full-image-onl

## [MINOR] batched_trainer.py:273
**Resume chi kiem tra state_dim va so action — doi env/reward config trong yaml roi resume se tron lan replay buffer cu (reward che do cu) voi config moi ma khong canh bao**

- Van de: Khoi resume (batched_trainer.py:273-300) chi validate 2 thu: state_dim (dong 275-279) va so luong action (280-285). resume.pt co luu day du 'train_cfg'/'env_cfg'/'state_cfg' (_save_resume_checkpoint, dong 124-126) nhung khong bao gio duoc doi chieu voi cfg/env_cfg hien tai lay tu yaml. Cac config cloud deu dat resume: true (ft_rl_cloud_s42.yaml:28) — neu giua 2 lan chay yaml bi sua (vd doi hard_hi
- Phan bien xac nhan: Doc code xac nhan quan sat cua reviewer LA DUNG ve mat su that: khoi resume (batched_trainer.py:273-304) chi raise khi state_dim (275-279) hoac so action (281-285) lech; train_cfg/env_cfg/state_cfg tuy duoc luu (124-126) nhung khong bao gio doc lai/doi chieu, va replay buffer cu duoc nap nguyen (291) roi optimize() (536) tinh Q-target bang gamma/reward_clip moi tren cac transition mang reward cu. 
- **FIX:** Khi resume, so sanh cac field reward/shaping trong resume_data['train_cfg']/['env_cfg'] (vd hard_hit_reward, crop_fp_penalty, target_reward, gamma, reward_clip, max_slices, use_crop_outcome_reward) voi cfg/env_cfg hien tai; neu lech thi in canh bao ro rang (hoac raise tru khi co co --allow-config-drift) thay vi im lang. Khong can check toan bo conf

## [MINOR] benchmark_speed.py:102
**Hai con số 'SAHI' (760ms tuần tự vs 608ms batched) đo bằng hai implementation khác nhau; benchmark_speed không chú thích tuần tự**

- Van de: benchmark_speed đo SAHI bằng _predict_fixed_sahi — chạy YOLO TỪNG crop tuần tự (vòng for trong eval/benchmark.py:153-163) và in nhãn chỉ là '+SAHI'; benchmark_oneshot đo cùng lưới 0.35/0.2 nhưng batch theo chunk và in nhãn 'SAHI (batched)'. Chênh 760 vs 608ms hoàn toàn là chênh implementation (batching), không phải chênh phương pháp, nhưng chỉ phía oneshot có chú thích.
- Phan bien xac nhan: Factually accurate but low-impact labeling nit, not a code/logic bug. Verified: benchmark_speed.py:102 calls _predict_fixed_sahi (eval/benchmark.py:152-168 runs YOLO per-crop sequentially in a for loop) and prints label "+SAHI" (line 143); benchmark_oneshot.py batches the identical 0.35/0.2 grid via batch_crops (line 191-192) and prints "SAHI (batched)" (line 274). Same grid + same merge_iou=0.5 m

## [MINOR] benchmark_speed.py:80
**small_thr (ngưỡng vật nhỏ) tính trên đúng subset --limit → small_recall không so sánh được giữa các run khác limit/split**

- Van de: Cả benchmark_speed.py:80 và benchmark_oneshot.py:82 tính small_thr = percentile 40 diện tích GT trên CHÍNH subset limit ảnh đang chạy. Đổi --limit (100 vs 150) hoặc --split (test vs val) là đổi định nghĩa 'vật nhỏ', nên small_recall giữa các run/scripts với limit khác nhau đo trên các tập GT-nhỏ khác nhau.
- Phan bien xac nhan: Doc code xac nhan: small_thr THUC SU tinh tren dung subset --limit. benchmark_speed.py:77 lay images = iter_images(split, limit) roi :80 truyen chinh list do vao _small_area_threshold, ham nay (benchmark.py:88-97) tra ve np.percentile(area_ratio GT cua CHINH cac anh do, 40). _evaluate_method:431 dung nguong nay lam small_mask, nen ca small_total lan small_hit -> small_recall doi dinh nghia khi doi
- **FIX:** Chot 1 quy uoc chung cho moi benchmark script khi dua vao Thesis: hoac dung CUNG --limit + CUNG --split cho tat ca run, hoac tinh small_thr MOT lan tren TOAN BO split (khong phu thuoc --limit) roi truyen vao _evaluate_method, de dinh nghia 'vat nho' (va do small_recall) dong nhat giua cac run/bang.
