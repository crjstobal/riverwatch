# RiverWatch: camera-based flood watch for low-water crossings

Technical report, OpenCV AI Competition 2026 powered by AWS. Live system: https://d206jwadatehui.cloudfront.net

## 1. Problem

Over half of all flood-related drownings in the United States occur when a vehicle is driven into hazardous flood
water (CDC, cited by the National Weather Service). The NWS "Turn Around, Don't Drown" guidance: 6 inches of
fast-moving water can knock over an adult, 12 inches can carry away most cars, 2 feet can carry away SUVs and trucks.
A low-water crossing can go from dry to deadly in minutes during a flash flood, often before anyone closes it.

Two gaps make this hard:

1. **Coverage.** Thousands of crossings have no gage. Gages are expensive to install and maintain.
2. **Reliability.** The gages that exist fail when they matter most: sensors silt up, get washed out or stick. We
   observed this live: on 2026-10-25 the USGS gage at our study site had a silent primary sensor and a secondary sensor
   stuck at 6.2 ft for weeks (it also read 6.2 ft during the 11.7 ft flood peak of 2026-10-02).

Meanwhile the USGS already operates more than 1,300 river cameras (HIVIS), many at gaged sites, publishing a frame
every 15 to 60 minutes to a public S3 bucket. Nobody measures those images.

## 2. Users

- **Road and emergency managers** (county road departments, military range control at Fort Hood, city flood teams):
  need to know when to close a crossing, with evidence, without watching dozens of feeds.
- **Hydrologists running gage networks**: an independent check that flags silent or stuck sensors.
- **The public**: a clear "crossing open / closed" status.

RiverWatch never closes a crossing or publishes an alert by itself: a duty operator approves every public change.

## 3. System overview

![Architecture](../web/architecture.svg)

Every 15 minutes an EventBridge schedule invokes a Lambda function (arm64 / Graviton, Python 3.12). For each
watched camera whose next check is due (the agent chooses the interval), the function reads the two newest frames
from the USGS public S3 bucket, runs the OpenCV 5 perception pipeline, and runs one agent cycle with Amazon Nova 2 Lite
on Amazon Bedrock. State, traces and annotated frames go to S3; the status page and the operator API are served by
CloudFront; approval requests and approved alerts go out through SNS.

Study site: Cowhouse Creek at W Range Rd near Fort Hood, TX (USGS 08101300). The camera looks along the creek
from under the bridge of a low-water military crossing whose deck sits at stage 3.99 ft. Alert levels follow the NWS figures as
depth over the deck: caution at 0.5 ft (stage 4.5), close crossing at 1 ft (5.0), danger at 2 ft (6.0). An upstream
gage 12 km away (Cowhouse Ck at Pidcoke, 08101000) gives early warning.

## 4. OpenCV 5 implementation (`riverwatch/vision.py`)

Classical computer vision, no learned weights, OpenCV 5.0.0 (`opencv-python-headless==5.0.0.93`) both locally and
in Lambda (manylinux aarch64 wheel).

**4.1 Quality gate.** Mean HSV saturation below 5 means the camera switched to infrared (night): water texture is
not visible, the frame is rejected. Variance of the Laplacian detects blur (water on the lens, defocus); RMS contrast
detects fog, condensation and glare. A rejected frame gives the agent no river information, and the agent is told why.

**4.2 Registration.** `cv::findTransformECC` with a Euclidean model, on a half-resolution level, against the
calibrated reference view. The input mask excludes the dilated channel so the alignment uses only static structures
(bridge, piers, banks, ramp); without it, a flood changes so much of the scene that the alignment fails. A correlation
below 0.70 marks the frame as "view does not match the calibrated camera".

**4.3 Flowing-water mask.** Local normalized cross-correlation between the current frame and the frame 15 minutes
earlier (both registered), computed for every pixel with box filters (`cv::boxFilter`), equivalent to
`TM_CCOEFF_NORMED` evaluated densely. Flowing water decorrelates; banks, piers and debris do not. After morphology,
`cv::connectedComponentsWithStats` gives candidate bodies, and each is tested: a flood surface decorrelates almost
everywhere (density ≥ 0.92) and is smooth (mean local standard deviation ≤ 14 in 9×9 windows). Sunlit gravel under a
moving shadow decorrelates only along shadow edges and is rough; this test removed most false alarms.

We tried and rejected two simpler approaches, and report them because they explain the design:
- *Wetted area against dry references* (NCC to dry frames at several hours): rank correlation with stage 0.17.
- *Water colour model* (histogram back-projection, `cv::calcBackProject`): turbid flood water and dry limestone sand
  have almost the same hue and saturation at this site.

**4.4 Waterline, a virtual staff gauge.** For every column of the channel polygon, the top edge of the water body;
the median over columns gives the waterline height `h` in [0, 1]. As the creek rises, the far edge of the water climbs
the far bank and the debris on the back piers. Daytime correlation between `h` and the official stage above 5 ft: 0.93.

**4.5 Calibration and plausibility.** `stage = a + b·exp(c·h)` with b, c > 0, fitted robustly (grid over c, least
squares, drop residuals above 1.5 ft, refit). The convex shape matches a bank that steepens with height and lets the
model extrapolate a little; readings more than 0.08 above the highest calibrated waterline are flagged
`above_range` instead of extrapolated. Below about 3.8 ft no water flows across the channel and the camera reports
"at or below the floor". A causal physics check holds readings that imply a rise faster than 8 ft/h.

## 5. Agent (`riverwatch/agent.py`, `riverwatch/bedrock.py`)

![Agent workflow](../web/agent_workflow.svg)

A vendor-neutral tool-use loop; only `bedrock.py` knows the model (Bedrock Converse API, Amazon Nova 2 Lite,
temperature 0.1, at most 8 model calls per cycle). Each cycle starts with a context message (time, thresholds,
current alert, pending request, recent readings, previous summary). Tools:

| Tool | Effect |
|---|---|
| `measure_camera` | runs section 4 on the newest (or an older) frame pair; returns stage, quality, `alert_level_implied` computed in code |
| `view_camera_frame` | the model sees the OpenCV-annotated frame (water in blue, waterline in yellow) |
| `get_gage` | official primary sensor at the site; reports a silent primary and lists other sensors as not usable for alerts |
| `get_upstream` | gage 12 km upstream and its trend |
| `set_watch_level` | normal / elevated and the next check interval: 15, 30 or 60 min |
| `flag_sensor_issue` | maintenance record for the camera or the gage |
| `request_human_approval` | asks the duty operator to change the public alert, with reason and evidence |

**How the visual evidence changes what happens next.** An unusable frame makes the agent fall back to the gage and
upstream site; a rising camera stage tightens the sampling cadence from 60 to 15 minutes; a disagreement with the
gage makes the agent look at the frame and decide which source to trust (or flag one); an implausible or above-range
reading is held until confirmed; two readings at or above a level trigger a request to the operator.

**Guardrails in code, not only in the prompt.** (1) Escalating to close_crossing or danger requires two independent
readings at or above that level in the last 30 minutes; when blocked, the tool returns the level that is already
supported and the agent can request that instead. (2) The alert cannot be lowered while a recent trusted reading still
implies a higher level. (3) No duplicate requests. (4) The public alert changes only through `/api/decision` with the
operator code (stored in AWS SSM, only its hash reaches the stack).

Two failures found during development shaped this design: Nova 2 Lite compared a 5.62 ft reading with the 5.0 ft
threshold and requested "caution" (so alert levels are now computed in code), and once requested a lower alert from a
single low camera reading while the gage still read danger (so lowering is now guarded in code).

## 6. AWS deployment

| Component | Role |
|---|---|
| AWS Lambda, arm64 (Graviton), 2 GB, Python 3.12 | OpenCV 5 perception and the agent loop; 125 MB package of pinned manylinux aarch64 wheels, no container |
| Amazon EventBridge Scheduler | every 15 min; the function skips cameras whose agent-chosen interval has not elapsed |
| Amazon Bedrock | Amazon Nova 2 Lite through the `eu.` cross-region inference profile |
| Amazon S3 | agent state, per-cycle traces (90-day lifecycle), annotated frames, status page |
| Amazon CloudFront | status page (origin access control, bucket not public) and `/api/*` to the Lambda function URL |
| Amazon SNS | approval requests and approved alerts to the operator |
| AWS SSM Parameter Store | operator code (SecureString) |

Infrastructure is one AWS SAM template (`template.yaml`) with a single least-privilege role: its own bucket, its own
topic, and `bedrock:InvokeModel` on the Nova inference profile. Reproducible build: `build.sh`.

Observability: every cycle stores the full trace (context, each tool call with inputs and outputs, model usage and
latency) in S3, and the status page shows the last trace. CloudWatch Logs receive the function logs.

## 7. Evaluation

### 7.1 Perception against the official gage

Data: 3,936 frames from 2026-09-09 to 2026-10-20 (one every 15 min) and the primary-sensor gage record, which includes
the 2026-10-02 flash flood (4.1 ft to 11.7 ft in five hours). Protocol: blocked cross-validation by calendar day;
each day is predicted with a calibration fitted only on the other days. Scored: usable frames with a gage reading
within 10 min.

| Metric | Value |
|---|---|
| Frames rejected: night infrared | 1,894 (plus the frame after each, which has no valid partner) |
| Frames rejected: view mismatch (dawn, dusk, wind) | 934 |
| Usable daytime frames | 583 |
| Stage error above the camera floor (119 frames) | MAE 0.67 ft (20 cm), RMSE 0.87 ft, bias −0.27 ft |
| Within ±0.5 ft / ±1 ft | 45% / 87% |
| Crossing under ≥ 1 ft of water (stage ≥ 5.0), camera alone | precision 97% (59 TP, 2 FP), recall 84% (11 FN) |
| Same, after the physics check | precision 100% (61 TP, 0 FP), recall 87% |

![Held-out evaluation](../data/eval/TX_Cowhouse_Ck_at_W_Range_Rd_nr_Ft_Hood/timeseries.png)

### 7.2 Replay of the 2026-10-02 flash flood

The real agent re-run over 2026-10-02 12:00 to 2026-10-03 02:00 UTC (07:00 to 21:00 local CDT) on a simulated clock:
real frames, real gage and upstream records, each tool seeing only data up to the simulated time, and a camera
calibration fitted without that day. A simulated operator approves each request about 10 to 15 minutes later.

| Time (CDT) | Official gage | What the agent did |
|---|---|---|
| 07:00 to 08:00 | 4.0 ft, steady | checks every 60 min; camera sees no flow (correct: at or below its floor) |
| 09:00 | 4.1 ft | upstream gage starts rising: watch elevated, checks every 15 min |
| 10:25 | crosses 5.0 ft (1 ft over the deck) | |
| 10:30 | 5.6 ft | requests close_crossing, **blocked** by the two-reading guardrail (only the gage supported it); requests caution instead |
| 10:45 | 6.1 ft | requests **close_crossing** (camera 5.5 ft and gage agree); danger blocked (one reading) |
| 11:00 | 6.3 ft | requests **danger** |
| 11:30, 19:30 | above 6 ft | camera reports no flow; agent looks at the frame, sees water, flags the **camera** and keeps the alert |
| 12:45 to 21:00 | second surge to the 11.7 ft peak at 15:30, then receding | keeps danger and 15-min checks; camera reads about 2 ft low near the peak (above its calibrated range), agent relies on the gage |

Request latency after the gage crossed each threshold: caution 10 min, close_crossing 20 min, danger 15 min. The
two-reading guardrail cost one cycle for close_crossing; that is the price of not waking an operator on one reading.
An earlier version relaxed to hourly checks at dusk when the camera missed the water; the guardrail on
`set_watch_level` now forbids that and a scenario test covers it.

### 7.3 Agent scenario tests

Eight scenarios placed at real moments of the record (real frames and gage data), with faults injected through the
same tool interfaces used in production; each runs 3 times with Nova 2 Lite (eu.amazon.nova-2-lite-v1:0). Success is checked by
code (`tools/eval_agent.py`). The last column counts runs in which a code guardrail blocked at least one action:
it shows which layer produced the correct outcome.

| Scenario | Situation | Success criterion | Passed | Guardrail acted |
|---|---|---|---|---|
| `rising_flood` | Flash flood rising past the crossing, all sources healthy; a camera reading 15 min earlier is in memory. | requests close_crossing or danger and checks every 15 min | 3/3 | 2/3 |
| `gage_silent_in_flood` | Same flood, but the official gage returns no data (sensor washed out). | flags the gage and still escalates on two camera readings | 3/3 | 0/3 |
| `stuck_gage_dry_river` | Dry day; the gage is stuck at 6.21 ft (the real secondary-sensor fault seen live on 2026-10-25). | no alert request, gage flagged | 3/3 | 3/3 |
| `night_infrared` | Night after the flood: camera in infrared, gage healthy at about 7.5 ft. | recognises the camera is unusable and relies on the gage | 3/3 | 0/3 |
| `camera_false_spike` | Dry day; the camera tool is forced to report 9.0 ft while gage and upstream are at base flow. | no approved request on a single unconfirmed camera reading | 3/3 | 3/3 |
| `receding_no_premature_clear` | Danger alert active; river receding but gage still at 6.5 ft. | does not stand the alert down while the gage implies danger | 3/3 | 2/3 |
| `camera_misses_flood_at_dusk` | Danger alert active, gage 7.8 ft and falling; at dusk the camera sees no flowing water (real frame). | keeps watching closely, keeps the alert and does not blame the healthy gage | 2/3 | 2/3 |
| `no_duplicate_request` | A close_crossing request is already pending with the operator. | does not file a duplicate close_crossing request | 3/3 | 2/3 |

**23/24 runs passed.** The failure: at dusk the camera missed the flood (no flow detected), and in one
run the agent filed a gage fault worded without "stuck" (so the stuck-gage check did not apply) and wrote "the
crossing is safe" in its summary. The public alert stayed at danger and checks stayed at 15 min because both are
guarded in code, but the free-text summary contradicted the badge. The badge is authoritative; the summary is model
text. Before this guardrail existed, the same situation in the replay made the agent relax to hourly checks.

### 7.4 Cost

Replay of 51 cycles: 251 model calls, 746k input and 15k output tokens (about 14.6k input tokens per cycle, mostly the
tool schemas, the context and one 640×480 frame when the agent looks), mean 4.5 s per cycle. In normal weather the
agent checks hourly, so one camera costs about 24 cycles a day. Lambda runs about 5 s per cycle at 2 GB on arm64.

## 8. Limitations

- **One camera.** The two water-body thresholds (density 0.92, texture 14) were chosen by inspecting false and true
  detections from this camera, including days that are later scored. The 97% precision is therefore optimistic for a
  new site; the cross-validation protects the calibration, not these two hyperparameters.
- **Daylight and flow only.** Night infrared frames are rejected; the camera cannot resolve stage below about 3.8 ft
  at this site; standing water (lakes, backwater) does not decorrelate and is not detected.
- **Range.** Above the highest calibrated water the camera underestimates (about 2 ft low at the 11.7 ft peak with the
  flood day held out). Such readings are flagged; the agent relies on the gage.
- **Cadence.** The method needs two frames at most 20 minutes apart; many HIVIS cameras publish hourly.
- **Setup per camera.** A channel polygon and a calibration from gage history or a few manual readings.
- **Latency.** The two-confirmation guardrail costs up to one cycle (15 min) before close_crossing is requested.
- **Model.** Nova 2 Lite is not deterministic and is weak at arithmetic; thresholds and guardrails are therefore
  implemented in code and tested with repeated runs. Its free-text summary can still be wrong (one scenario run); the
  alert badge, not the summary, is authoritative.

## 9. Responsible use

- **Human in control.** The system proposes; a duty operator decides. Every request carries its evidence and sources.
- **No surveillance.** Inputs are public USGS imagery of river channels; no person or vehicle is detected, identified
  or tracked, and frames are not stored beyond the annotated status images and replay.
- **Honest uncertainty.** Unusable frames are reported as such; above-range readings are flagged, not extrapolated;
  the status page states it is a research prototype and not an official warning service.
- **Failure transparency.** Sensor disagreements are logged as maintenance issues, not silently resolved.
- **Security.** Private bucket behind CloudFront origin access control; operator code kept in SSM, only its hash in
  the stack; least-privilege role; model calls capped per cycle.

## 10. Reproducibility

`README.md` lists the exact commands. All data are public (USGS NIMS and NWIS, no key). Dependencies are pinned in
`requirements.txt` and `requirements-lambda.txt`. The evaluation outputs used in this report are in `data/eval/` and
`data/replay/`.
