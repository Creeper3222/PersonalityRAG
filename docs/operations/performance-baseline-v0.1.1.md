# PersonalityRAG v0.1.1 performance baseline

Baseline commit: `f92cf7f2caac4c3ebae0c49a424c3ffc733ce1aa`.

The measurements below were collected from the Windows live process with two
libraries before the v0.1.1 operational performance work. They are diagnostic
baselines and must be compared on the same machine and data snapshot.

| Endpoint/resource | Mean | Observed p95 | Total/wall |
| --- | ---: | ---: | ---: |
| `GET /api/v1/libraries?stats_mode=summary` | 27.01 ms | - | - |
| 20 concurrent library summaries | 212.06 ms | - | 260.37 ms |
| warm `GET /api/v1/stats` | - | 94.23 ms | - |

The live process reported 68 threads and 259.63 MiB working set at the sample
point. The performance target is at most 100 ms mean and 150 ms wall time for
20 concurrent summary reads, at most 40 ms warm `/stats` p95, and no sustained
thread growth after connection concurrency.

Use the checked-in read-only benchmark for repeatable HTTP samples:

```powershell
.\.venv\Scripts\python.exe tools\benchmark_operations.py `
  --rounds 10 --concurrency 20 `
  --output data\reports\performance-v0.1.1.json
```

The tool reads the API credential from the local state configuration but never
includes it in console or report output. Mutating operation benchmarks use
isolated test state and temporary data; they must not target production data.

## Final Windows live result

The optimized Windows live build at `791eda7` was measured after an application
restart on the same machine and production data snapshot. The first cold run
included one `/stats` cache fill at `115.34 ms`; the immediately repeated warm
run produced the acceptance values below.

| Endpoint/resource | Mean | p95 | Total/wall |
| --- | ---: | ---: | ---: |
| `GET /api/v1/libraries?stats_mode=summary` | 11.07 ms | - | - |
| 20 concurrent library summaries | 28.90 ms | 40.18 ms | 46.90 ms |
| warm `GET /api/v1/stats` | - | 26.24 ms | - |

The live process held at 30 threads before and after two benchmark rounds. Its
working set moved from 135.65 MiB to 140.89 MiB without continued thread growth.
Reports are written to `data/reports/performance-v0.1.1-final.json` and
`data/reports/performance-v0.1.1-final-repeat.json` in the ignored runtime data
tree.
