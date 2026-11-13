# Outreach emails (drafts) — review before sending

Repo: https://github.com/kruuusher13/gptnl-energy-estimation-ekf
Models: https://huggingface.co/kruuusher13/gptnl-energy-models

Quick try (both):
```
pip install git+https://github.com/kruuusher13/gptnl-energy-estimation-ekf.git
gptnl-energy forecast --n 400000
gptnl-energy monitor --test --gate-demo
```

---

## 1. To Julio de Oliveira Filho (julio.deoliveirafilho@tno.nl)

**Subject:** The energy predictor is live: code, models, and the math

Hi Julio,

I wanted to share this with you first, because it turned into exactly the tool we talked about. The whole thing is now public and installable, and I think you will enjoy poking at it.

The short version: you run one command before launching a curation job and get the predicted energy, cost, and CO2 with a confidence interval, and while the job runs a live monitor refines the estimate from the EAR telemetry. This is the measure-and-estimate approach from Section 4.1.2 of the architecture document, actually built and validated.

Since you wanted to see the math, here is where it lives:

1. Each stage gets a two-parameter energy law E = c0 + c1*n, calibrated from a few small runs, with a closed-form prediction interval. The whole method rests on one condition I make precise in the thesis: you have to sample above the "overhead floor" n* = c0/c1, otherwise the slope is not identifiable. Above it, the dominant stage predicts a 400k-document run at 2.8% error.
2. The live estimator is an Extended Kalman Filter reduced to its measurement update (the stages are sequential, so there is no state transition to model). Its innovation gate is what makes it robust on shared nodes: a contaminated reading that would push the estimate +93% off gets rejected and the error stays at +8%.
3. The result I did not expect: summing the raw per-task EAR records of a parallel run overstates the true energy by 2 to 11 times, because each record charges a whole node's power to a task that only partly occupies it. It is in the thesis with the figure. I would really like your read on whether that matches what the SURF side sees.

Two minutes to try it:

    pip install git+https://github.com/kruuusher13/gptnl-energy-estimation-ekf.git
    gptnl-energy forecast --n 400000
    gptnl-energy monitor --test --gate-demo

The second command replays a recorded run and shows the gate catching a contaminated reading. Everything (code, the 442 measurements, the trained models, the thesis) is in the repo, and every number regenerates from the scripts in paper/code.

Repo: https://github.com/kruuusher13/gptnl-energy-estimation-ekf
Models: https://huggingface.co/kruuusher13/gptnl-energy-models

I would love to hear what you think, and where you would want to take it next. Thanks for pointing me at this problem in the first place.

Best,
Romir

---

## 2. To Martino Mensio

**Subject:** My thesis tool is public: energy prediction for the GPT-NL curation pipeline

Hi Martino,

Quick note to share that the thesis work is finished and, more importantly, that it shipped as a working tool rather than staying on paper. I would be glad if you gave it a spin.

The idea in one line: predict the energy of a full data curation run from a handful of short trial runs, before committing the multi-day job. It fits a simple physics model per pipeline stage, forecasts the full run with a confidence interval, and tracks the estimate live while the job executes. Across four corpora spanning a 100x range in document length, the dominant stage predicts at 2.8% error, and a small transfer model beats every neural network I trained at generalizing to a corpus it never saw, which was a nice validation that the physics-first approach was the right call here.

If you want to see it run:

    pip install git+https://github.com/kruuusher13/gptnl-energy-estimation-ekf.git
    gptnl-energy forecast --n 400000

The repo has the thesis, the full code, the measurement data, and the trained models, and everything is reproducible from the analysis scripts.

Repo: https://github.com/kruuusher13/gptnl-energy-estimation-ekf
Models: https://huggingface.co/kruuusher13/gptnl-energy-models

Thank you for the guidance along the way. Happy to walk you through any part of it.

Best,
Romir
