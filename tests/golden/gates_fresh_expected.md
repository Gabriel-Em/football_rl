# Gate Report (tag=self_sampled, gens=100,200,300, protocol=fresh, calibration=calibration.json, calibration_gen=300)

| Gate | Description | Status |
|---|---|---|
| C0 | catastrophe floors (gens >= 200) | FAIL |
| C0L | learning floor (T2 gain from gen 100 to the last gen) | FAIL |
| G1 | ball holding (relative) | PASS |
| G2 | keeper (relative) | PASS |
| G3 | clumping (relative) | FAIL |
| G4 | role order (relative) | PASS |
| G5 | scoring (relative) | PASS |
| G6 | takeoff (share of seeds without takeoff) | PASS |

## C0 Catastrophe Check

| Seed | Gen | Metric | Rule | Value |
|---|---|---|---|---|
| 44 | 200 | scoreless | > 0.70 | 0.7500 |

## C0L Learning Floor

FAIL if T2(300) - T2(100) < 0.50.

| Seed | T2 at 100 | T2 at 300 | Gain | Status |
|---|---|---|---|---|
| 42 | -3.000 | -1.000 | +2.000 | PASS |
| 43 | -3.000 | -2.800 | +0.200 | FAIL |
| 44 | -3.000 | -1.500 | +1.500 | PASS |
| 45 | -3.000 | -1.200 | +1.800 | PASS |

## Takeoff (G6)

| Seed | Takeoff gen | History gens |
|---|---|---|
| 42 | 25 | 1-30 |
| 43 | 27 | 1-30 |
| 44 | 25 | 1-30 |
| 45 | none | 1-30 |

Seeds without takeoff: 1 of 4 (share 0.250; FAIL if > 0.5).

## Relative Guards (G1-G5 at gen 300)

| Gate | Metric | Worse | Arm seeds | Anchor seeds | Mean diff | Delta | Status |
|---|---|---|---|---|---|---|---|
| G1 | hold_share_pooled | lower | 42=0.250, 43=0.260, 44=0.270, 45=0.280 | 42=0.200, 43=0.220, 44=0.210, 45=0.190 | +0.0600 | 0.0258 | PASS |
| G1 | holder_spell_s_pooled | lower | 42=0.340, 43=0.350, 44=0.360, 45=0.330 | 42=0.300, 43=0.320, 44=0.310, 45=0.290 | +0.0400 | 0.0258 | PASS |
| G2 | gk_in_box_pooled | lower | 42=0.980, 43=0.985, 44=0.975, 45=0.980 | 42=0.980, 43=0.985, 44=0.975, 45=0.980 | +0.0000 | 0.0100 | PASS |
| G2 | gk_goal_line_pooled | higher | 42=0.150, 43=0.160, 44=0.170, 45=0.140 | 42=0.150, 43=0.170, 44=0.160, 45=0.140 | +0.0000 | 0.0300 | PASS |
| G3 | n_within_100_pooled | higher | 42=0.800, 43=0.850, 44=0.820, 45=0.830 | 42=0.500, 43=0.550, 44=0.520, 45=0.530 | +0.3000 | 0.0416 | FAIL |
| G3 | cluster_ge4_pooled | higher | 42=0.050, 43=0.060, 44=0.050, 45=0.040 | 42=0.050, 43=0.060, 44=0.055, 45=0.045 | -0.0025 | 0.0129 | PASS |
| G4 | role_inversion_pooled | higher | 42=0.100, 43=0.110, 44=0.120, 45=0.100 | 42=0.100, 43=0.120, 44=0.110, 45=0.090 | +0.0025 | 0.0300 | PASS |
| G5 | goals_total | lower | 42=2.000, 43=2.100, 44=2.200, 45=1.900 | 42=2.000, 43=2.200, 44=2.100, 45=1.900 | +0.0000 | 0.3000 | PASS |
| G5 | scoreless | higher | 42=0.100, 43=0.120, 44=0.110, 45=0.090 | 42=0.100, 43=0.120, 44=0.110, 45=0.090 | +0.0000 | 0.0500 | PASS |

## Between-seed SD at gen 300

| Gate | Metric | Arm SD | Comp SD |
|---|---|---|---|
| G1 | hold_share_pooled | 0.0129 | 0.0129 |
| G1 | holder_spell_s_pooled | 0.0129 | 0.0129 |
| G2 | gk_in_box_pooled | 0.0041 | 0.0041 |
| G2 | gk_goal_line_pooled | 0.0129 | 0.0129 |
| G3 | n_within_100_pooled | 0.0208 | 0.0208 |
| G3 | cluster_ge4_pooled | 0.0082 | 0.0065 |
| G4 | role_inversion_pooled | 0.0096 | 0.0129 |
| G5 | goals_total | 0.1291 | 0.1291 |
| G5 | scoreless | 0.0129 | 0.0129 |

## Targets

| Target | Gen | Dir | Delta min | Arm seeds | Comp seeds | Mean diff | Verdict | Note |
|---|---|---|---|---|---|---|---|---|
| T1 (informational at gen 300) | 300 | higher | 2.582 | 42=-6.000, 43=-7.000, 44=-5.000, 45=-6.500 | 42=-10.500, 43=-11.500, 44=-10.000, 45=-12.000 | +4.875 | BEATS |  |
| T2 (primary) | 300 | higher | 0.516 | 42=-1.000, 43=-2.800, 44=-1.500, 45=-1.200 | 42=-1.400, 43=-1.000, 44=-1.800, 45=-0.900 | -0.350 | NEUTRAL |  |
| T3 (kick_outcome_teammate@self_sampled) | 300 | higher | 0.030 | 42=0.310, 43=0.300, 44=0.320, 45=0.310 | 42=0.300, 43=0.310, 44=0.320, 45=0.290 | +0.005 | NEUTRAL |  |

## Style Targets

| ID | Key | Dir | Floor | Delta | Arm seeds | Comp seeds | Mean diff | Verdict |
|---|---|---|---|---|---|---|---|---|
| S1 | hold_share_pooled | higher | 0.01 | 0.0258 | 42=0.250, 43=0.260, 44=0.270, 45=0.280 | 42=0.200, 43=0.210, 44=0.190, 45=0.180 | +0.0700 | BEATS |
| S2 | aim_teammate_share_sampled | higher | 0.02 | 0.0258 | 42=0.100, 43=0.120, 44=0.110, 45=0.130 | 42=0.100, 43=0.130, 44=0.110, 45=0.120 | +0.0000 | NEUTRAL |
| S3 | kick_outcome_opponent | lower | 0.02 | 0.0258 | 42=0.550, 43=0.560, 44=0.540, 45=0.570 | 42=0.440, 43=0.460, 44=0.450, 45=0.470 | +0.1000 | LOSES |
| S4 | goal_no_pass_10s_share | lower | 0.05 | 0.0516 | 42=0.770, 43=0.810, 44=0.790, 45=0.800 | 42=0.780, 43=0.800, 44=0.820, 45=0.760 | +0.0025 | NEUTRAL |

## Per-team Informational (gen 300)

| Metric | Arm mean | Per seed |
|---|---|---|
| hold_share_A | 0.1330 | 42=0.130, 43=0.132, 44=0.134, 45=0.136 |
| hold_share_B | 0.1230 | 42=0.120, 43=0.122, 44=0.124, 45=0.126 |
| gk_in_box_A | 0.9820 | 42=0.985, 43=0.983, 44=0.981, 45=0.979 |
| gk_in_box_B | 0.9770 | 42=0.980, 43=0.978, 44=0.976, 45=0.974 |
| gk_goal_line_A | 0.1650 | 42=0.150, 43=0.160, 44=0.170, 45=0.180 |
| gk_goal_line_B | 0.1350 | 42=0.120, 43=0.130, 44=0.140, 45=0.150 |
| cluster_ge4_A | 0.0600 | 42=0.060, 43=0.060, 44=0.060, 45=0.060 |
| cluster_ge4_B | 0.0400 | 42=0.040, 43=0.040, 44=0.040, 45=0.040 |
| role_inversion_A | 0.1100 | 42=0.110, 43=0.110, 44=0.110, 45=0.110 |
| role_inversion_B | 0.0900 | 42=0.090, 43=0.090, 44=0.090, 45=0.090 |