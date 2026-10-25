# Devpost submission: RiverWatch

**Project name:** RiverWatch

**Tagline (max ~200 chars):** OpenCV 5 reads the water level from existing USGS river cameras; an Amazon Nova agent cross-checks gages, flags broken sensors and asks a human to close flooded low-water crossings.

**Prize tracks:** Overall; Agentic Vision Award.

**Links**
- Live system: https://d206jwadatehui.cloudfront.net
- Code: REPO_URL
- Video: VIDEO_URL
- Technical report: REPO_URL/blob/main/docs/report.md

## Inspiration

Over half of flood drownings in the US happen when someone drives into flood water, and low-water crossings are
where that happens. On 2 October 2026 a military crossing near Fort Hood, Texas, went from dry to 7 ft under water in
five hours. The USGS camera there filmed all of it. Nobody was measuring the images.

## What it does

RiverWatch turns river cameras into water-level sensors and puts an agent in charge of what happens next.

- **Sees:** OpenCV 5 measures the waterline on every frame pair and converts it to stage (ft), with a quality gate
  that refuses to guess at night, in fog or when the camera has moved.
- **Reasons:** an Amazon Nova agent compares the camera with the official gage and the gage 12 km upstream, can look
  at the annotated frame itself, and decides how often to check (15, 30 or 60 min).
- **Acts, with a human:** it files sensor issues and asks the duty operator to close the crossing, with evidence.
  Guardrails in code require two independent readings to escalate and forbid lowering an alert while a reading still
  implies danger. Nothing is published without the operator.

Live today it reports the crossing as dry and flags that the official gage is broken: its primary sensor is silent
and the backup is stuck at 6.2 ft.

## How we built it

- **OpenCV 5.0.0**, classical vision only: Laplacian / HSV / contrast quality gate; `findTransformECC` registration
  masked to static structures; dense local normalized cross-correlation (box filters) between frames 15 min apart to
  find flowing water; per-body density and smoothness tests (`connectedComponentsWithStats`) to reject sunlit gravel
  under moving shadows; waterline as a virtual staff gauge with a robust monotone calibration and a physics check.
- **Agent:** vendor-neutral tool-use loop on the Bedrock Converse API with Amazon Nova 2 Lite; 8 tools; guardrails
  and alert levels computed in code; every cycle traced.
- **AWS:** Lambda on arm64 Graviton (OpenCV 5 from pinned manylinux aarch64 wheels, no container), EventBridge
  Scheduler, Bedrock, S3, CloudFront (origin access control), SNS, SSM Parameter Store; one SAM template.
- **Data:** USGS HIVIS cameras (public S3 bucket) and NWIS gages, public domain.

## Evaluation

- 3,936 frames, blocked cross-validation by day (the flood day never seen during calibration).
- When water flows: MAE 0.67 ft (20 cm), 87% of readings within ±1 ft of the official gage.
- "Crossing under ≥ 1 ft of water" from the camera alone: precision 97%, recall 84%.
- Agent: 23/24 scenario runs passed (silent gage, stuck gage, night, false camera spike, receding flood, missed
  detection at dusk, duplicate request), plus a full replay of the 2 October flood on a simulated clock.
- Failure cases documented: night infrared, sunlit gravel, readings above the calibrated range, missed water at dusk.

## Challenges

- Turbid flood water and dry limestone have nearly the same colour, so colour models failed; motion between frames
  worked, but moving shadows on gravel produced false alarms until we tested each water body for density and texture.
- A flood changes most of the image, which broke whole-frame registration; masking ECC to static structures fixed it.
- The gage at our site publishes three series, and one is stuck: using it blindly made the agent raise a false
  danger alert on the first live run. Now only the primary sensor counts and the others are reported as unreliable.
- The model compared 5.62 with 5.0 wrongly and once relaxed the watch on a single bad camera reading; both are now
  enforced in code and covered by scenario tests.

## What we learned

Put the arithmetic and the safety rules in code, give the model the evidence (including the picture), and test the
agent like a system: scenarios with injected faults, repeated runs, and a replay of a real event on a simulated clock.

## What's next

More cameras (the HIVIS network has more than 1,300), a time-of-day background model for shadows, calibration from
a few manual readings for ungauged crossings, and integration with county road-closure systems.

## Built with

opencv, python, numpy, amazon-bedrock, amazon-nova, aws-lambda, aws-graviton, amazon-s3, amazon-cloudfront, amazon-sns, amazon-eventbridge, aws-sam, usgs
