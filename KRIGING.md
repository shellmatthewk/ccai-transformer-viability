# Ordinary kriging baseline

Run `scripts\run_kriging.cmd` after or alongside the neural evaluations. Kriging
does not train for epochs. It selects an exponential-covariance length scale
from 200, 400, 800, and 1600 km using 60 fixed 2018 validation days, then uses
2019 for conformal calibration and 2020-2021 for final testing.

The system uses ordinary-kriging weights with an unknown constant mean, a five
percent nugget, and a common Europe-centered equirectangular projection. It
receives the same nested 256/192/128/64/26 station contexts and target samples
as the learned models. Use `kriging_experiment.py` as the authoritative entry
point; it applies the common geographic projection to contexts and targets.
