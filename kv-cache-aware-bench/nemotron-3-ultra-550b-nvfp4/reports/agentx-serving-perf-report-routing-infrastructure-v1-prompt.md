# AgentX routing infrastructure illustration

[View the illustration](agentx-serving-perf-report-routing-infrastructure-v1.png).

Generated with the built-in image_gen tool, using an initial generation and two edits. This is a conceptual illustration for review; cache contents and load indicators are illustrative. The aggregated fleet represents six SGLang workers with four GB300 GPUs each; the disaggregated alternative represents eight prefill and eight decode workers with four GPUs per worker.

## Initial generation

```text
Use case: infographic-diagram.
Create one vivid, polished infrastructure illustration for a Google Cloud technical blog. Wide landscape canvas, high resolution, exceptionally readable typography. It should feel like a beautifully designed cloud-console UI combined with subtle 3D GPU/server illustrations: white and very pale blue surfaces, rounded cards, gentle shadows, crisp navy labels, vivid blue and violet request paths, green capacity/status accents. Strong visual hierarchy and generous spacing. This is an educational systems diagram, not a screenshot of a real product. All labels must be correctly spelled, short and easy to read.

Title: "Agentic inference on Google Cloud"
Subtitle: "NVIDIA Dynamo routes requests using prefix reuse + worker load"

MAIN COMPOSITION — clear left-to-right story, occupying most of the canvas.
LEFT, about 23% width: a panel titled "AIPerf AgentX replay". Small dataset archive icon labeled "Weka 256K traces" and "393 session roots". Below, show TWO recorded session trees, "Session A" in blue and "Session B" in violet. Each has a root, successive turn cards, and small branching subagent cards with joining lines. Colored token strips show a repeated blue or violet prefix followed by a few orange new-input blocks. A clock icon represents recorded delays. Small terminal/repository icons represent recorded coding activity, not live tools. Labels: "Shared prefix", "New input", "Recorded delays + subagent dependencies". Both sessions send inference requests toward the router.

CENTER AND RIGHT sit inside a clearly visible, spacious boundary labeled "Google Kubernetes Engine (GKE)".
CENTER, about 25% width: a prominent "Dynamo frontend" card containing a "KV-aware router" hub. Two compact information cards feed the hub: "Cache index" with subtitle "Prefix → worker locations", and "Active-load tracker" with subtitle "Assigned prefill / decode work". Below the hub, a concise decision label: "Select using reuse + load". Draw the router as a vivid intelligent routing hub with clean connectors, not a humanoid robot. Actual cached tensors belong to the workers, not inside the router.

RIGHT, about 50% width: a fleet titled "Aggregated serving · 24 GPUs" and subtitled "6 SGLang workers × 4 GB300 GPUs". EXACTLY SIX worker cards in a 2-row × 3-column grid, labeled sequentially Worker 1, Worker 2, Worker 3 on the top row and Worker 4, Worker 5, Worker 6 on the bottom row. Every card visibly contains FOUR small GPU-chip tiles, a "KV cache" shelf with colored prefix blocks, and a qualitative load meter. Label chip group "4 × GB300".
Worker 1: blue "Prefix A" cache, "Busy".
Worker 2: blue "Prefix A" cache, "Available", highlighted with a blue/green selection border.
Worker 3: violet "Prefix B" cache, "Busy".
Worker 4: gray "Other prefix" cache, "Available".
Worker 5: violet "Prefix B" cache, "Available", highlighted with a violet/green selection border.
Worker 6: gray "Other prefix" cache, "Moderate load".
Strong blue request arrow from the router ends clearly at Worker 2. Strong violet request arrow ends clearly at Worker 5. Do not send the highlighted paths to other worker cards. These are two illustrative placements: matching cached context plus available capacity. Add a compact callout beside the workers: "Reuse valid prefix → Prefill new input → Decode response". Another callout: "High load can outweigh cache reuse".
Use a neatly routed thin dashed gray cache-event bus from worker-cache shelves back toward the cache index, labeled "Cache insertion / eviction metadata". Keep this visually distinct from request arrows. Add a slim return path labeled "Streamed responses" toward the replay client, placed along the bottom of the main system area. Keep connectors orderly with minimal crossings.

BOTTOM INSET, clearly separated and subordinate: title "Disaggregated alternative · 64 GPUs". Within it show two distinct miniature worker pools: "8 prefill workers" → an arrow labeled "State transfer" → "8 decode workers". Show eight tiny worker tiles in each pool, with label "4 GPUs per worker". Under prefill: "Prefix reuse + load". Under decode: "Decode-load selection". This inset is an alternative serving topology, not additional workers in the 24-GPU main fleet.

FOOTER LEGEND: blue/violet blocks = "Shared context"; orange blocks = "New input"; solid arrows = "Requests / responses"; dashed gray arrows = "Cache metadata".
Small footer note: "Cache contents and load meters are illustrative. AIPerf replays inference traffic from recorded sessions."

Constraints: preserve the exact six-worker main fleet and 24-GPU total; distinguish the 64-GPU disaggregated inset. Show real infrastructure character through GPU tiles, worker pods, cache shelves and routing connectors. No invented throughput, latency, cache-hit percentages or utilization numbers. No claim that the benchmark executes live coding tools. Do not imply routing sends the full model or KV tensors through the cache index. No wall of equations, no excessive prose, no fake buttons, no watermarks. Render a coherent, vivid and technically credible UI-style illustration with beautiful detail and legible labels.
```

## First edit

```text
Edit the attached infrastructure diagram. Preserve its overall layout, vivid UI style, readable labels, six main worker cards (Worker 1–6), all four GPU chips per main worker, session-tree panels, and Dynamo router. Make this targeted technical-accuracy and legibility pass:

1. Correct the outbound REQUEST connectors. The BLUE request route must terminate with a visible arrowhead touching the highlighted MIDDLE TOP card labeled "Worker 2". The VIOLET request route must terminate with a visible arrowhead touching the highlighted MIDDLE BOTTOM card labeled "Worker 5". Currently they incorrectly point into Worker 1 and Worker 4. Remove those incorrect connector ends entirely. Route the replacement lines cleanly through the open gutters around the cards; do not draw through GPU icons or labels. Label each route clearly. It is essential that arrowheads physically end at Worker 2 and Worker 5, not merely have text mentioning them.
2. Correct cache metadata direction. Thin dashed gray connectors originate at worker cache shelves and feed a shared bus, which ends at the "Cache index" card in the Dynamo frontend. The arrow at the frontend must point INTO "Cache index", not into "Active-load tracker". Remove downward arrowheads that imply the index sends cache events to workers. Preserve label "Cache insertion / eviction metadata".
3. The lower inset must contain EXACTLY EIGHT small prefill server icons and EXACTLY EIGHT decode server icons. The current prefill panel has nine; reduce it to eight. Preserve "8 prefill workers", "8 decode workers", "4 GPUs per worker", "State transfer", and "Disaggregated alternative · 64 GPUs".
4. Replace the dark shadow/gradient behind the top title with a clean pale blue-white background so the navy title and subtitle have strong contrast. Preserve the title text. Keep the Google Cloud logo, but remove the tiny unsolicited slogan beneath it.
5. The streamed response return path should originate from a small shared response collector below the selected Worker 2 and Worker 5 routes, rather than being attached only to Worker 6.

Do not change the illustrated worker cache states, load labels, target-worker highlights, or fleet totals. Keep other parts intact. No invented performance numbers. The finished diagram must remain attractive and coherent, with clean orthogonal connectors and clear directionality.
```

## Final edit

```text
Refine this image by SIMPLIFYING its connections and correcting the worker placement. Keep the vivid light cloud-console UI style, readable title, left AgentX replay panel, central Dynamo frontend and six worker cards with four GPU chips each.

Rearrange the SIX worker cards into this EXACT spatial order:
TOP ROW left to right:
  Worker 2 — Prefix A — Available — green highlighted border
  Worker 1 — Prefix A — Busy — normal border
  Worker 3 — Prefix B — Busy — normal border
BOTTOM ROW left to right:
  Worker 5 — Prefix B — Available — green highlighted border
  Worker 4 — Other prefix — Available — normal border
  Worker 6 — Other prefix — Moderate load — normal border

This deliberately puts both selected workers in the LEFTMOST COLUMN, immediately next to the router. Make the short BLUE request arrow from the router terminate directly at top-left Worker 2. Make the short VIOLET request arrow from the router terminate directly at bottom-left Worker 5. Delete the blue arrow running horizontally across another worker card and every old request-arrow segment that targets any other card. The top-middle and bottom-middle cards must NOT be highlighted.

REMOVE the confusing gray dashed cache metadata connector network above the worker cards entirely. Add a plain small caption inside the Cache index card: "Updated by worker cache events". Keep the existing Cache index → router and Active-load tracker → router input arrows. Do not add any new dashed lines to workers or to the load tracker. Remove the dashed-arrow item from the footer legend.

REMOVE the invented "Response collector" box and its database icon entirely. Instead draw one slim return arrow from the outer worker-fleet boundary, below the worker grid, back to the replay client, labeled "Streamed responses". This is a conceptual response path with no extra server component. Do not attach it specifically to Worker 4 or Worker 6.

SIMPLIFY the lower disaggregated inset. Replace the rows of tiny individual server icons with ONE attractive server-pool pictogram per group. The LEFT group label must read "8 prefill workers" and "4 GPUs per worker", with "Prefix reuse + load" beneath. The RIGHT group label must read "8 decode workers" and "4 GPUs per worker", with "Decode-load selection" beneath. One arrow between pools labeled "State transfer". The single pictogram represents each whole pool, so do not draw eight or nine individual icons. Keep total "64 GPUs".

Keep main fleet title "Aggregated serving · 24 GPUs" and subtitle "6 SGLang workers × 4 GB300 GPUs". Keep the footer note that states are illustrative and AIPerf replays recorded inference traffic. Preserve all other explanatory content, clean bright title background and Google Cloud branding. Prioritize correct physical arrow endpoints and a clean, easy-to-follow layout.
```
