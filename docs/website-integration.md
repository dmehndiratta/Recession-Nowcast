# Website integration (one-time, by Dhruv)

Pattern A (SETUP.md §3–§8). The cross-repo workflow keeps
`Website/public/recession-nowcasting/` fresh automatically; you only do the
**one-time** hand edit below, plus add the secrets.

## Secrets (project repo → Settings → Secrets and variables → Actions)
- `WEBSITE_REPO_TOKEN` — **required**. Fine-grained PAT, resource owner
  `dmehndiratta`, repository access **only** `Website`, permission
  *Contents: Read and write*.
- `FRED_API_KEY` — optional. The pipeline falls back to FRED's keyless CSV
  endpoints if absent. StatCan and BoC are keyless.

## One-time `/research` card (add to the listing in `D:/Website`)

```html
<article class="research-card">
  <span class="status live">LIVE</span>
  <div class="tags">
    <span>MACRO</span><span>CAPITAL MARKETS</span><span>TIME SERIES</span><span>NOWCASTING</span>
  </div>
  <h3><a href="/research/recession-nowcasting">Recession Nowcasting</a></h3>
  <p>Real-time, calibrated US &amp; Canada recession probabilities from data as it was
     actually available at each date (no look-ahead), with uncertainty bands and an
     honest out-of-sample horse-race against the term-spread probit and the Sahm rule.</p>
  <a class="readmore" href="/research/recession-nowcasting">→ Read more</a>
  <a class="gh" href="https://github.com/dmehndiratta/Recession-Nowcast">GitHub</a>
</article>
```

## One-time page `src/pages/research/recession-nowcasting.mdx`

> If the site uses a content collection (`src/content/research/`) instead of
> `src/pages/research/`, move the frontmatter into the collection schema and keep
> the body — see SETUP.md §9 open question 4.

```mdx
---
title: "Recession Nowcasting"
status: "LIVE"
tags: ["MACRO", "CAPITAL MARKETS", "TIME SERIES", "NOWCASTING"]
date: "2026-06"
abstract: "Calibrated real-time recession probabilities with uncertainty bands, US & Canada."
---

import IframeEmbed from "../../components/IframeEmbed.astro";

# Recession Nowcasting

Real-time recession probabilities built only from data as it was available at each
historical date, benchmarked out-of-sample against the term-spread probit and the
Sahm rule, and served as a probability **band** rather than a single number.

<IframeEmbed src="/recession-nowcasting/dashboard.html" title="Recession nowcast dashboard" />

[Open the dashboard full screen](/recession-nowcasting/dashboard.html) ·
[Read the full report](/recession-nowcasting/report.html) ·
[GitHub](https://github.com/dmehndiratta/Recession-Nowcast)

## What this is
A mixed-frequency dynamic factor model (with penalised-MIDAS and gradient-boosted
comparators) evaluated in a pseudo-real-time expanding window, with block-bootstrap
intervals, Diebold–Mariano / Giacomini–White tests, calibration curves, and a
vintage-vs-final refutation. Updated weekly.
```

## Verify the deploy
1. Trigger `workflow_dispatch` on `recession-nowcast-update`; confirm both jobs green.
2. Open `https://dhruv-mehndiratta.com/recession-nowcasting/dashboard.html` — the
   probability band renders and `/recession-nowcasting/data/*.json` parses.
3. Confirm a later scheduled run updates the "last updated" stamp.
