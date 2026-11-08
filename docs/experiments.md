# Experiments

Mean absolute error (MAE) in %MVC (percent of maximal force) on the 16 evaluation people, mean over fingers,
recordings and days, then people, unless stated. A − B: mean difference, 95 % bootstrap confidence interval over
people, and how many people are lower with A. Written by `python -m experiments.report`.

## New day, no labels from it

Trained on the person's day 1 and both days of 19 other people, scored on the five day-2 trials. Both ridges
use running-mean inputs. The person-weighted one gives the person the TCN's share of each batch. MAE at rest:
frames with the label below 5 %MVC.

| model | MAE ± SD | R² | r | MAE at rest | − ridge, person-weighted |
|---|---|---|---|---|---|
| zero force | 9.83 ± 3.16 | -0.19 | – | 1.99 | +1.13 (+0.73 to +1.57), 0/16 |
| ridge | 9.08 ± 2.82 | 0.01 | 0.52 | 3.90 | +0.38 (+0.13 to +0.60), 2/16 |
| ridge, person-weighted | 8.70 ± 2.99 | 0.08 | 0.60 | 4.03 | – |
| LSTM (5 seeds) | 7.45 ± 2.80 | 0.15 | 0.69 | 4.01 | -1.25 (-1.65 to -0.82), 15/16 |
| TCN (5 seeds) | 7.09 ± 2.77 | 0.28 | 0.72 | 3.74 | -1.62 (-2.06 to -1.12), 15/16 |

Trained on the person's day 1 only (5 seeds): 9.89. Final − that: -2.81 (-3.82 to -2.00), 16/16.

Never trained on the person (the new-person TCN, seed 0): 7.99 on the same recordings.
Adding their day 1 (seed 0): -0.94 (-1.66 to -0.29), 12/16.

Labels use each day's own maximal-force (MVC) recordings. A user would have only day 1's: rescored with them
(seed 0), the TCN scores 6.64, the person-weighted ridge 8.22 and zero force 9.38.

## New person, no data from them

Trained on both days of 19 other people. TCN: seed 0.

| model | MAE ± SD | R² | r | MAE at rest | − ridge |
|---|---|---|---|---|---|
| zero force | 9.37 ± 2.68 | -0.24 | – | 1.96 | +0.55 (+0.29 to +0.81), 3/16 |
| ridge | 8.82 ± 2.53 | -0.11 | 0.45 | 3.56 | – |
| TCN | 7.67 ± 1.84 | -0.01 | 0.61 | 4.01 | -1.16 (-1.63 to -0.69), 15/16 |

## What each change contributed

Same people, seed 0.

New day:

| recipe | MAE | − previous |
|---|---|---|
| TCN on the person's day 1 (running-mean inputs) | 9.82 | – |
| + 19 other people, electrode shifts and CAR | 7.66 | -2.17 (-2.80 to -1.65), 16/16 |
| − running mean (final recipe) | 7.05 | -0.61 (-1.08 to -0.13), 12/16 |

New person:

| recipe | MAE | − previous |
|---|---|---|
| TCN on 19 other people (running-mean inputs) | 8.41 | – |
| + electrode shifts and CAR | 8.14 | -0.27 (-0.53 to -0.03), 10/16 |
| − running mean (final recipe) | 7.67 | -0.48 (-0.76 to -0.21), 12/16 |

## Same day against new day

Each person's TCN trains without their day-1 trials 4-5 and is scored on those and on day-2 trials 4-5.
Constant = the median of the person's own training labels. Day-2 trials contain more force, hence the ratio.

| scored on | TCN MAE | constant MAE | TCN / constant |
|---|---|---|---|
| same day (day-1 trials 4-5) | 4.85 | 9.23 | 0.54 |
| new day (day-2 trials 4-5) | 7.29 | 10.10 | 0.72 |

New day − same day: +2.45 (+1.20 to +3.88), higher for 14/16.

## A held press

Hyser presses last a few seconds. Each person's model decodes 2 s of day-2 rest, then the trial's strongest 1 s
repeated 12 times. Ratio = output 10-11 s into the hold / output 1-2 s into it (1 = holds), per finger pressing
at least 10 %MVC.

| TCN inputs | median ratio | presses ≥ 0.7 | people with median ≥ 0.7 | presses used (early output ≥ 2 %MVC) |
|---|---|---|---|---|
| no running mean (final) | 1.00 | 100% | 6/6 | 35/39 |
| running mean (previous) | 0.40 | 15% | 0/6 | 34/39 |

## Timing and strong presses

Shift = the delay of the decoded force (±500 ms, 10 ms steps) that best correlates with the label, per day-2
recording and moving finger. Negative = ahead. Peak = the highest decoded force in a press of at least 100 ms
at 20 %MVC or more, over the measured one.

| model | shift (ms), median over people | decoded / measured peak | people with low peaks |
|---|---|---|---|
| TCN | -25 | 0.68 (0.60 to 0.75) | 15/16 |
| ridge, person-weighted | -105 | 0.30 (0.26 to 0.35) | 16/16 |

## Labelled calibration on the new day

The frozen new-day TCN fine-tuned (all layers, learning rate 1e-4, 100 steps), against ridges refitted on day 1
plus the same labelled data. Seed 0. First seconds of day-2 trial 1, scored on trials 2-5:

| method | 0 s | 5 s | 10 s | 25 s |
|---|---|---|---|---|
| TCN, frozen | 7.17 | – | – | – |
| TCN fine-tuned | – | 7.37 | 7.19 | 6.22 |
| ridge refit, no normalisation | 13.23 | 9.67 | 8.68 | 7.66 |
| ridge refit, running mean | 9.68 | 9.32 | 9.19 | 8.86 |

Whole day-2 trials 1 to 3, scored on trials 4-5:

| method | 0 s | 25 s | 50 s | 75 s |
|---|---|---|---|---|
| TCN, frozen | 7.19 | – | – | – |
| TCN fine-tuned | – | 6.37 | 5.61 | 5.10 |
| ridge refit, no normalisation | 13.15 | 7.95 | 7.02 | 6.53 |
| ridge refit, running mean | 9.69 | 8.92 | 8.54 | 8.29 |

Fine-tuned at 75 s − frozen: -2.09 (-2.91 to -1.38), 16/16.

## Input monitor

A channel whose log-RMS falls more than 3 units below or rises more than 3 above its usual
offset from its grid's median is replaced by median + offset. Non-finite samples become 0. A whole grid going
silent is not detected. Faults start 5 s into each day-2 recording, MAE after the fault.

| fault from 5 s | MAE, clean | MAE, monitor off | MAE, monitor on | flagged within |
|---|---|---|---|---|
| a row of 8 electrodes reads zero | 7.32 | 7.39 | 7.38 | 170 ms (median) |
| one channel stuck at a constant | 7.32 | 7.43 | 7.35 | 225 ms (median) |
| one channel picks up noise at 100x its level | 7.32 | 13.18 | 7.46 | 10 ms (median) |
| 100 ms of NaN on every channel | 7.32 | 7.37 | 7.34 | as non-finite input |

Clean recordings: 0.016% of channel-frames replaced, in 41 of 80 recordings.
MAE 7.05 with the monitor off (as scored everywhere else), 7.01 on.

## Latency

The TCN in `assets/` on Apple M1, one thread, batch 1, on recorded EMG. An update = filter, framing, input
monitor and network. Paced runs sleep between updates like a timer-driven loop. p50 and p99: median of the three
runs. Max: over all three. Late: done more than 10 ms after its deadline.

| timed | p50 ms | p99 ms | max ms | late, of all updates |
|---|---|---|---|---|
| network only (ONNX) | 0.103 | 0.110 | 0.206 | – |
| whole update, back to back (ONNX) | 0.346 | 0.393 | 0.577 | – |
| whole update, paced every 10 ms (ONNX) | 2.197 | 2.784 | 23.799 | 16 of 17787 |
| whole update, paced every 10 ms (PyTorch) | 2.943 | 4.218 | 7.629 | 3 of 17787 |

## Simulated gripper

Decoded thumb and index flexion (day-2 trial 1) set a 1-15 N grip, tracked by a proportional-integral (PI) force
controller on a fixed object in MuJoCo. Errors against the grip set from the force label. Under- and over-grip =
more than 1 N below or above it.

| decoder | decoder MAE (N) | controller tracking MAE (N) | whole chain MAE (N) | time under-grip | time over-grip |
|---|---|---|---|---|---|
| ridge (person's day 1 only) | 1.55 | 0.12 | 1.56 | 30% | 14% |
| TCN, person's day 1 only | 1.70 | 0.18 | 1.72 | 28% | 17% |
| TCN, final recipe | 1.22 | 0.19 | 1.22 | 22% | 14% |

Whole chain, final − day-1-only TCN: -0.49 N (-0.78 to -0.27), 16/16.

## How each setting was chosen

On the 4 development people (1, 8, 10, 12), new day, development MAE. 0.30 rule = at least 0.30 %MVC lower on
the mean and lower for at least 3 of the 4 people. Those runs are kept outside the repo.

| setting | tried | rule | kept |
|---|---|---|---|
| TCN regularisation | dropout 0.1, weight decay 1e-4, no normalisation: 14.08 | 0.30 rule | dropout 0.3, weight decay 1e-2, running mean: 9.15 |
| other people's data | person's day 1 only: 9.15 | lowest mean, set after the first recipe was scored on the 16 | + 19 other people, half of each batch from the person: 8.46 |
| electrode-shift augmentation | none: 8.46 | the same | shifts up to 1 electrode: 8.24 |
| electrodes | 64: 8.30 | the same | 256: 8.24 |
| input normalisation | running mean: 8.24, held press 0.23 | holds a press (ratio ≥ 0.7), at most 0.30 worse | none: 7.66, held press 1.00 |
| training length and dropout | 1000 to 4000 steps, dropout 0.1 or 0.3: 7.67 to 8.10 | change only if 0.30 lower | 2000 steps, dropout 0.3: 7.66 |
| network | causal LSTM: 8.20 | lower mean, holds a press, streams in under 10 ms | TCN: 7.80 (3 seeds each, lower for 4 of 4) |
| spatial montage | as recorded: 7.80, row differences: 8.33 | 0.30 rule | CAR: 7.46. Rerun without the evaluation people's day 2: 7.72 against 7.78, kept, the 16 had already been scored with it |
| calibration | head-only, longer or faster fine-tuning: 7.21 to 8.17 | lowest mean over 5, 10 and 25 s | all layers, learning rate 1e-4, 100 steps: 7.00 |

## What didn't help

MAE or change in MAE (%MVC), on the recipe of the time. Each was worse, or better by less than 0.30 %MVC. Code
kept outside the repo.

| idea | tested on | result | compared with |
|---|---|---|---|
| 640 ms window instead of 320 ms | the 16, seed 0 | +0.01 (−0.07 to +0.11) | TCN |
| per-finger output gain, 25 s labelled | the 16 | −0.14 (−0.30 to −0.01) | frozen TCN |
| warm-up mean, or running mean and SD, as input normalisation | development, ridge | +0.05 to +0.65 | running mean |
| slow exponential mean (30 s) or rest tracker as normalisation | development | 10.87 and 11.16 | none 7.71 |
| channel dropout on top of electrode shifts | development, new person | 9.99 | shifts only 9.73 |
| random channel masks in ridge training | development | 9.56 | ridge 9.81 |
| day-2 input alignment: per-channel z-score or CORAL | development | 9.59 | ridge 9.81 |
| electrode-shift estimation from activation maps | development | 9.78 | ridge 9.81 |
| non-negative factorisation, components refitted on day 2 | development | 10.21 | ridge 9.81 |
| motor-unit decoding with day-1 separation filters | development | worse than ridge for 2 of 3 | ridge |
| input standardisation re-estimated on unlabelled day-2 EMG | development | 8.97 | frozen TCN 8.48 |
| alignment of the frozen TCN's internal features | development | 8.88 | frozen TCN 8.48 |
| per-channel input scale and bias, 25 s labelled | development | 7.95 | fine-tune 7.80 |
| learned 256 × 256 input map, 25 s labelled | development | 7.60 | fine-tune 7.80 |
| 5 × 5 output re-mix, 75 s labelled | the 16 | +0.11, better for 7 of 16 | frozen TCN |
| average of the frozen TCN and a CAR ridge | the 16 | +0.38, better for 4 of 16 | frozen TCN |

