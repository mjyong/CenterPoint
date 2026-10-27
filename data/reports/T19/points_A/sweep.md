# Parameter sweep

- scenes: 0921_164127_s000, 0921_164127_s001, 0921_164127_s002, 0921_164127_s003, 0921_164127_s004, 0921_164127_s005, 0921_165659_s003, 0921_165659_s004, 0921_165659_s005, 0921_165659_s006
- detections: `raw` from `data/tune/deploy/{scene}/pillar/results.pkl`
- every row against row 1; ahead: share of 500 paired resamples of the scenes in which the row is better

Columns: det recall = detections: recall within near range; det precision = detections: precision within near range; det F1 = detections: F1 within near range; det FP / frame = detections: false per keyframe; MOTA = tracks: MOTA; IDF1 = tracks: IDF1; IDSW / min = tracks: ID switches / min; false tracks / min = tracks: false tracks / min; confirm p90 s = tracks: confirm delay p90 [s]; moving missed = motion: relevant moving stretches missed; false moving / h = motion: parked reported moving / h; FDE 3 s = prediction: FDE 3 s, moving [m]; MR 3 s = prediction: miss rate 3 s, moving.

| # | parameters | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=5 | 0.839 | 0.825 | 0.832 | 0.643 | 0.572 | 0.772 | 2.069 | 38.066 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 2 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=3 | 0.839 | 0.824 | 0.832 | 0.652 | 0.571 | 0.771 | 2.069 | 37.860 | 1.200 | 2 | 86.891 | 2.795 | 0.521 |
| 3 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=8 | 0.836 | 0.824 | 0.830 | 0.614 | 0.580 | 0.775 | 2.069 | 35.170 | 1.200 | 2 | 86.891 | 2.796 | 0.521 |
| 4 | tracker.high_score=0.4, tracker.output_min_score=0.35, tracker.low_score=0.2, filter.min_points.vehicle=12 | 0.832 | 0.827 | 0.829 | 0.597 | 0.589 | 0.778 | 2.069 | 32.067 | 1.500 | 2 | 86.891 | 2.809 | 0.530 |

Difference to row 1 (ahead %):

| # | det recall | det precision | det F1 | det FP / frame | MOTA | IDF1 | IDSW / min | false tracks / min | confirm p90 s | moving missed | false moving / h | FDE 3 s | MR 3 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.001 (67%) | -0.000 (23%) | 0.000 (51%) | 0.009 (0%) | -0.002 (8%) | -0.001 (8%) | 0.000 (0%) | -0.207 (57%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) |
| 3 | -0.002 (0%) | -0.000 (0%) | -0.001 (0%) | -0.029 (100%) | 0.008 (100%) | 0.003 (100%) | 0.000 (0%) | -2.896 (100%) | 0.000 (0%) | 0 (0%) | 0.000 (0%) | 0.001 (0%) | 0.000 (0%) |
| 4 | -0.007 (0%) | 0.002 (97%) | -0.002 (22%) | -0.047 (100%) | 0.017 (100%) | 0.006 (100%) | 0.000 (0%) | -6.000 (100%) | 0.300 (0%) | 0 (0%) | 0.000 (0%) | 0.014 (22%) | 0.009 (0%) |
