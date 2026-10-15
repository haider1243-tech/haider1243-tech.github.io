(() => {
  const P = window.PROFILE;
  const { esc, rich, duration, byNewest } = window.Site;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const TYPE_LABEL = { work: "Work", education: "Education", project: "Project", milestone: "Milestone" };
  const AREA_LABEL = { software: "Software", engineering: "Engineering", career: "Career" };
  const STATUS = { done: ["Complete", "cyan"], active: ["In progress", "amber"], planned: ["Planned", ""] };

  const ICONS = {
    plc: '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M7 12h10M12 7v10"/>',
    screen: '<rect x="3" y="4" width="18" height="13" rx="1.5"/><path d="M8 20h8"/>',
    clock: '<circle cx="12" cy="12" r="8"/><path d="M12 8v4l3 2"/>',
    network: '<path d="M4 18l6-12 4 8 3-5 3 9" stroke-linejoin="round"/>',
    bolt: '<path d="M13 2 4 14h6l-1 8 9-12h-6l1-8z" fill="currentColor" stroke="none"/>',
    target: '<circle cx="12" cy="12" r="3"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>',
    wave: '<path d="M4 12h4l2-6 4 12 2-6h4" stroke-linejoin="round"/>',
    chart: '<path d="M4 20V4h16"/><path d="M8 16l4-6 3 4 3-6"/>',
    draw: '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/>',
    cube: '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="12" cy="12" r="4"/>',
    code: '<path d="M8 7l-5 5 5 5M16 7l5 5-5 5M14 4l-4 16"/>'
  };
  const icon = (k) => `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">${ICONS[k] || ICONS.plc}</svg>`;

  // Everything that can reference a skill, for "used in" links.
  const LOG = P.log.filter((e) => e.site !== false).sort(byNewest);
  const usedIn = (skill) => [
    ...P.software.filter((s) => (s.skills || []).includes(skill)).map((s) => ({ label: s.name, go: () => goTo("#sw-" + s.id) })),
    ...LOG.filter((e) => (e.skills || []).includes(skill)).map((e) => ({ label: e.title, go: () => openLog(e.id) }))
  ];

  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => t.classList.remove("show"), 2200);
  }

  function goTo(sel) {
    const el = $(sel);
    if (!el) return;
    el.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });
    el.classList.remove("flash");
    void el.offsetWidth;
    el.classList.add("flash");
  }

  /* ---------- THEME ---------- */
  const themeBtn = $("#themeToggle");
  function applyTheme(t) {
    document.documentElement.dataset.theme = t;
    themeBtn.textContent = t === "light" ? "☾" : "☀";
    themeBtn.setAttribute("aria-label", t === "light" ? "Switch to dark theme" : "Switch to light theme");
    $('meta[name="theme-color"]').setAttribute("content", t === "light" ? "#F3F6FB" : "#0A0F1C");
  }
  applyTheme(document.documentElement.dataset.theme || "dark");
  themeBtn.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    applyTheme(next);
    try { localStorage.setItem("theme", next); } catch (e) {}
  });

  /* ---------- NAV: mobile menu, progress, active section ---------- */
  const menuBtn = $("#menuBtn"), navLinks = $("#navLinks");
  const setMenu = (open) => {
    navLinks.classList.toggle("open", open);
    menuBtn.setAttribute("aria-expanded", open);
    menuBtn.textContent = open ? "✕" : "☰";
  };
  menuBtn.addEventListener("click", () => setMenu(!navLinks.classList.contains("open")));
  navLinks.addEventListener("click", (e) => { if (e.target.closest("a")) setMenu(false); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") setMenu(false); });

  const bar = $("#scrollProgress");
  const onScroll = () => {
    const h = document.documentElement;
    const max = h.scrollHeight - h.clientHeight;
    bar.style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + "%";
  };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const links = new Map($$('a[href^="#"]', navLinks).map((a) => [a.getAttribute("href").slice(1), a]));
  const spy = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (!en.isIntersecting) return;
      links.forEach((a) => a.classList.remove("active"));
      const a = links.get(en.target.id);
      if (a) a.classList.add("active");
    });
  }, { rootMargin: "-45% 0px -50% 0px" });
  $$("main section[id]").forEach((s) => spy.observe(s));

  /* ---------- ABOUT: specs + stats ---------- */
  $("#specList").innerHTML = [
    ["LOCATION", P.location],
    ["CREDENTIAL", "EET Diploma, SAIT — April 2026"],
    ["DESIGNATION", P.designation],
    ["WORK STATUS", P.workStatus],
    ["FOCUS AREAS", "Protection · Controls · Automation"],
    ["SEEKING", "Technologist · Controls · Testing · Field service"],
    ["LANGUAGES", P.languages]
  ].map(([k, v]) => `<div class="spec-row"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`).join("");

  const allSkills = P.skills.flatMap((g) => g.items);
  const milestones = P.goals.flatMap((g) => g.milestones);
  const msDone = milestones.filter((m) => m.state === "done").length;
  const stats = [
    { n: new Date().getFullYear() - P.startYear, label: "Years in engineering" },
    { n: allSkills.filter((s) => !s.learning).length, label: "Tools used hands-on" },
    { n: P.certs.filter((c) => c.kind === "cert").length, label: "Certifications & registrations" },
    { n: msDone, suffix: ` / ${milestones.length}`, label: "Goal milestones hit" }
  ];
  $("#stats").innerHTML = stats.map((s) =>
    `<div class="stat"><div class="num"><span data-count="${s.n}">${reduceMotion ? s.n : 0}</span>${s.suffix ? `<small>${esc(s.suffix)}</small>` : ""}</div><div class="lbl">${esc(s.label)}</div></div>`
  ).join("");

  /* ---------- SKILLS ---------- */
  $("#skillsGroups").innerHTML = P.skills.map((g, gi) => `
    <div class="reveal" style="--chip-c:var(--${esc(g.color)})">
      <div class="skill-group-title">${esc(g.group)}</div>
      <div class="chip-row" data-group="${gi}">
        ${g.items.map((s) => {
          const n = usedIn(s.name).length;
          return `<button type="button" class="chip${s.learning ? " learning" : ""}" data-skill="${esc(s.name)}" aria-pressed="false">${icon(s.icon)}${esc(s.name)}${n ? `<span class="uses" title="Used in ${n} pathway entr${n > 1 ? "ies" : "y"}">${n}</span>` : ""}</button>`;
        }).join("")}
      </div>
      <div class="skill-caption" data-caption-for="${gi}" aria-live="polite"></div>
    </div>`).join("");

  function showSkill(chip, toggle = true) {
    const row = chip.closest(".chip-row");
    const caption = $(`.skill-caption[data-caption-for="${row.dataset.group}"]`);
    const already = chip.classList.contains("active");
    $$(".chip", row).forEach((c) => { c.classList.remove("active"); c.setAttribute("aria-pressed", "false"); });
    if (already && toggle) { caption.classList.remove("show"); return; }
    chip.classList.add("active");
    chip.setAttribute("aria-pressed", "true");
    const name = chip.dataset.skill;
    const skill = allSkills.find((s) => s.name === name);
    const uses = usedIn(name);
    caption.innerHTML = `${esc(skill.desc)}${uses.length ? `<div class="used-in">Used in: ${uses.map((u, i) => `<a role="button" tabindex="0" data-use="${i}">${esc(u.label)}</a>`).join("")}</div>` : ""}`;
    $$("[data-use]", caption).forEach((a) => {
      const go = () => uses[+a.dataset.use].go();
      a.addEventListener("click", go);
      a.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
    });
    caption.classList.add("show");
  }
  $$(".chip").forEach((chip) => chip.addEventListener("click", () => showSkill(chip)));

  function activateSkill(name) {
    const chip = $$(".chip").find((c) => c.dataset.skill === name);
    if (!chip) return;
    showSkill(chip, false);
    chip.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });
    chip.focus({ preventScroll: true });
  }
  const tagHTML = (names) => names.length
    ? `<div class="tag-row">${names.map((n) => `<button type="button" class="tag" data-skill-tag="${esc(n)}">${esc(n)}</button>`).join("")}</div>` : "";
  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-skill-tag]");
    if (t) activateSkill(t.dataset.skillTag);
  });

  /* ---------- CAPSTONE ---------- */
  const cap = P.log.find((e) => e.id === "capstone");
  if (cap) $("#capstoneList").innerHTML = cap.details.map((d) => `<li>${rich(d)}</li>`).join("");
  const capBtn = $("#capstoneToggle"), capBody = $("#capstoneBody");
  capBtn.addEventListener("click", () => {
    const open = capBody.classList.toggle("open");
    capBtn.setAttribute("aria-expanded", open);
    $(".label", capBtn).textContent = open ? "Hide technical breakdown" : "View technical breakdown";
  });

  /* ---------- SOFTWARE ---------- */
  const msHTML = (list) => `<ul class="ms-list">${list.map((m) =>
    `<li class="${esc(m.state)}"><span class="ms-check" aria-hidden="true">${m.state === "done" ? "✓" : ""}</span><span>${rich(m.text)}<span class="sr-only">${m.state === "done" ? " (done)" : m.state === "active" ? " (in progress)" : ""}</span></span></li>`).join("")}</ul>`;

  $("#softwareList").innerHTML = P.software.map((s) => {
    const [label, color] = STATUS[s.status] || STATUS.planned;
    const subject = encodeURIComponent(`${s.name} — early access`);
    const access = s.launched && s.url
      ? `<a class="btn btn-primary btn-sm" href="${esc(s.url)}" target="_blank" rel="noopener">Open ${esc(s.name)} ↗</a>`
      : `<a class="btn btn-primary btn-sm" href="mailto:${esc(P.email)}?subject=${subject}">Request early access</a>
         <span class="sw-lock">🔒 No public link yet</span>`;
    return `
    <article class="sw-card spot reveal" id="sw-${esc(s.id)}" itemscope itemtype="https://schema.org/SoftwareApplication">
      ${s.launched ? "" : `<div class="sw-banner"><span class="sw-banner-dot"></span>${esc(s.availability || "In development")}</div>`}
      <div class="sw-main">
        <div class="sw-head"><h3 itemprop="name">${esc(s.name)}</h3><span class="pill blue" itemprop="softwareVersion">${esc(s.version)}</span><span class="pill ${color}">${label}</span></div>
        <p class="sw-tagline" itemprop="description">${rich(s.tagline)}</p>
        <p class="sw-desc">${rich(s.desc)}</p>
        <div class="sw-tabs" role="tablist" aria-label="${esc(s.name)} details">
          <button type="button" role="tab" class="sw-tab" aria-selected="true" data-tab="features">Features</button>
          ${s.releases?.length ? '<button type="button" role="tab" class="sw-tab" aria-selected="false" data-tab="log">Build log</button>' : ""}
        </div>
        <div class="sw-panel" data-panel="features" role="tabpanel">
          <ul class="bullet-list">${s.features.map((f) => `<li>${rich(f)}</li>`).join("")}</ul>
        </div>
        ${s.releases?.length ? `
        <div class="sw-panel" data-panel="log" role="tabpanel" hidden>
          <div class="term" aria-label="Release history">
            <div class="term-bar"><i></i><i></i><i></i><span>${esc(s.id)} — git log --releases</span><button type="button" class="term-replay">↻ replay</button></div>
            <pre class="term-body" data-releases='${esc(JSON.stringify(s.releases))}'></pre>
          </div>
        </div>` : ""}
        ${s.note ? `<p class="sw-note">${esc(s.note)}</p>` : ""}
        <div class="sw-access">${access}</div>
      </div>
      <aside class="sw-side">
        <div class="side-title">Product roadmap</div>
        ${msHTML(s.roadmap)}
        <div class="side-title">Built with</div>
        ${tagHTML(s.skills || [])}
      </aside>
    </article>`;
  }).join("");

  // Build-log terminal: types the releases out, one line at a time.
  function playTerm(pre) {
    const rel = JSON.parse(pre.dataset.releases);
    const lines = rel.map((r) => `<span class="t-v">${esc(r.v)}</span> <span class="t-d">${esc(r.date)}</span>  ${esc(r.note)}`);
    const final = `<span class="t-ok">✓ ${rel.length} releases · next: public launch</span>`;
    clearTimeout(pre._t);
    if (reduceMotion) { pre.innerHTML = [...lines, final].join("\n"); return; }
    pre.innerHTML = "";
    let i = 0;
    const next = () => {
      if (i < lines.length) { pre.innerHTML += (i ? "\n" : "") + lines[i++]; pre.scrollTop = pre.scrollHeight; pre._t = setTimeout(next, 260); }
      else { pre.innerHTML += "\n" + final + '<span class="t-cursor">▋</span>'; }
    };
    next();
  }
  $$(".sw-card").forEach((card) => {
    const tabs = $$(".sw-tab", card);
    tabs.forEach((tab) => tab.addEventListener("click", () => {
      tabs.forEach((t) => t.setAttribute("aria-selected", t === tab));
      $$(".sw-panel", card).forEach((p) => { p.hidden = p.dataset.panel !== tab.dataset.tab; });
      const pre = $(".term-body", card);
      if (tab.dataset.tab === "log" && pre && !pre.dataset.played) { pre.dataset.played = 1; playTerm(pre); }
    }));
    const replay = $(".term-replay", card);
    if (replay) replay.addEventListener("click", () => playTerm($(".term-body", card)));
  });

  /* ---------- FILTER HELPER ---------- */
  function makeFilters(container, items, keyOf, labels, onChange) {
    const counts = {};
    items.forEach((i) => { counts[keyOf(i)] = (counts[keyOf(i)] || 0) + 1; });
    const keys = Object.keys(labels).filter((k) => counts[k]);
    container.innerHTML = [["all", "All", items.length], ...keys.map((k) => [k, labels[k], counts[k]])]
      .map(([k, l, n]) => `<button type="button" class="filter" data-filter="${k}" aria-pressed="${k === "all"}">${esc(l)}<span class="count">${n}</span></button>`).join("");
    const set = (k) => {
      $$(".filter", container).forEach((b) => b.setAttribute("aria-pressed", b.dataset.filter === k));
      onChange(k);
    };
    container.addEventListener("click", (e) => { const b = e.target.closest(".filter"); if (b) set(b.dataset.filter); });
    return set;
  }

  /* ---------- LOGBOOK ---------- */
  $("#logList").innerHTML = LOG.map((e) => {
    const dur = duration(e);
    const hasBody = e.details?.length || e.skills?.length || e.summary;
    return `
    <article class="log-entry type-${esc(e.type)}${e.end === "present" ? " ongoing" : ""}" id="log-${esc(e.id)}" data-type="${esc(e.type)}">
      <span class="log-dot" aria-hidden="true"></span>
      <div class="log-card spot">
        <button class="log-head" type="button" aria-expanded="false" aria-controls="logb-${esc(e.id)}"${hasBody ? "" : " disabled"}>
          <div>
            <div class="log-meta"><span class="pill">${TYPE_LABEL[e.type] || esc(e.type)}</span><span><time datetime="${esc(e.start)}">${esc(window.Site.fmtDate(e.start))}</time>${e.end ? ` – ${e.end === "present" ? "Present" : `<time datetime="${esc(e.end)}">${esc(window.Site.fmtDate(e.end))}</time>`}` : ""}</span>${dur ? `<span>· ${esc(dur)}</span>` : ""}${e.end === "present" ? '<span class="pill green">Ongoing</span>' : ""}</div>
            <h4>${esc(e.title)}</h4>
            <div class="log-org">${esc(e.org)}${e.location ? ` · ${esc(e.location)}` : ""}</div>
          </div>
          <span class="chevron" aria-hidden="true">+</span>
        </button>
        <div class="log-body" id="logb-${esc(e.id)}" hidden>
          ${e.summary ? `<p>${rich(e.summary)}</p>` : ""}
          ${e.details?.length ? `<ul class="bullet-list">${e.details.map((d) => `<li>${rich(d)}</li>`).join("")}</ul>` : ""}
          ${tagHTML(e.skills || [])}
        </div>
      </div>
    </article>`;
  }).join("") + '<p class="log-empty" id="logEmpty" hidden>Nothing here yet.</p>';

  function setLogOpen(entry, open) {
    entry.classList.toggle("open", open);
    $(".log-head", entry).setAttribute("aria-expanded", open);
    $(".log-body", entry).hidden = !open;
  }
  $$(".log-entry").forEach((entry) => {
    $(".log-head", entry).addEventListener("click", () => setLogOpen(entry, !entry.classList.contains("open")));
  });
  // Open the most recent entry so the section doesn't look empty.
  const first = $(".log-entry");
  if (first) setLogOpen(first, true);

  const setLogFilter = makeFilters($("#logFilters"), LOG, (e) => e.type, TYPE_LABEL, (k) => {
    let shown = 0;
    $$(".log-entry").forEach((el) => { el.hidden = !(k === "all" || el.dataset.type === k); if (!el.hidden) shown++; });
    $("#logEmpty").hidden = shown > 0;
  });

  function openLog(id) {
    const entry = $("#log-" + id);
    if (!entry) return;
    if (entry.hidden) setLogFilter("all");
    setLogOpen(entry, true);
    goTo("#log-" + id);
  }

  /* ---------- LESSONS ---------- */
  const lessons = [...P.challenges].sort((a, b) => byNewest({ start: a.date }, { start: b.date }));
  $("#lessonList").innerHTML = lessons.map((c) => `
    <article class="lesson spot" data-area="${esc(c.area)}">
      <div class="lesson-top"><span>${esc(c.project)} · ${esc(window.Site.fmtDate(c.date))}</span><span class="pill ${c.area === "software" ? "blue" : c.area === "engineering" ? "amber" : "cyan"}">${AREA_LABEL[c.area] || esc(c.area)}</span></div>
      <h4>${esc(c.title)}</h4>
      <dl>
        <dt class="p">Problem</dt><dd>${rich(c.problem)}</dd>
        <dt class="f">Fix</dt><dd>${rich(c.fix)}</dd>
      </dl>
      ${c.lesson ? `<p class="takeaway">“${esc(c.lesson)}”</p>` : ""}
    </article>`).join("");
  makeFilters($("#lessonFilters"), lessons, (c) => c.area, AREA_LABEL, (k) => {
    $$(".lesson").forEach((el) => { el.hidden = !(k === "all" || el.dataset.area === k); });
  });

  /* ---------- GOALS ---------- */
  const pct = (list) => list.length ? Math.round((list.filter((m) => m.state === "done").length / list.length) * 100) : 0;
  const overall = pct(milestones);
  $("#goalOverall").innerHTML = `
    <div><div class="big">${overall}%</div><div class="desc">Overall pathway</div></div>
    <div class="bar" role="progressbar" aria-valuenow="${overall}" aria-valuemin="0" aria-valuemax="100" aria-label="Overall progress"><span data-w="${overall}"></span></div>
    <div class="desc">${msDone} of ${milestones.length} milestones · ${P.goals.filter((g) => g.status === "active").length} goals in progress</div>`;

  $("#goalGrid").innerHTML = P.goals.map((g) => {
    const [label, color] = STATUS[g.status] || STATUS.planned;
    const p = pct(g.milestones);
    return `
    <article class="goal spot ${esc(g.status)}">
      <div class="goal-phase"><span>PHASE <b>${esc(g.phase)}</b> — ${esc(g.when.toUpperCase())}</span><span class="pill ${color}">${label}</span></div>
      <h4>${esc(g.title)}</h4>
      <p>${rich(g.desc)}</p>
      <div class="goal-progress"><div class="bar" role="progressbar" aria-valuenow="${p}" aria-valuemin="0" aria-valuemax="100" aria-label="${esc(g.title)} progress"><span data-w="${p}"></span></div><span>${p}%</span></div>
      ${msHTML(g.milestones)}
      <div class="goal-meta">${esc(g.area)}</div>
    </article>`;
  }).join("");

  /* ---------- CERTS ---------- */
  $("#certGrid").innerHTML = P.certs.map((c) =>
    `<div class="cert-card spot${c.featured ? " featured" : ""}"><div class="name">${esc(c.name)}</div><div class="issuer">${esc(c.issuer)}</div></div>`).join("");

  /* ---------- CONTACT ---------- */
  $("#contactLinks").innerHTML = `
    <a href="mailto:${esc(P.email)}" class="btn btn-primary">${esc(P.email)}</a>
    <button type="button" class="btn btn-outline" id="copyEmail">Copy email</button>
    <a href="tel:${esc(P.phoneHref)}" class="btn btn-outline">${esc(P.phone)}</a>
    <a href="${esc(P.linkedin)}" class="btn btn-outline" target="_blank" rel="noopener">LinkedIn ↗</a>
    <a href="${esc(P.github)}" class="btn btn-outline" target="_blank" rel="noopener">GitHub ↗</a>
    <a href="resume.html" class="btn btn-outline">Resume ↓</a>`;
  $("#copyEmail").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(P.email); toast("Email copied ✓"); }
    catch (e) { toast(P.email); }
  });

  $("#footer").innerHTML = `${esc(P.name)} — ${esc(P.location)} · Built with care, one relay at a time. · Last updated ${esc(window.Site.fmtDate(P.updated))} · <a href="resume.html">Resume</a>`;

  /* ---------- REVEAL + COUNTERS + BARS ---------- */
  function countUp(el) {
    const target = +el.dataset.count;
    if (reduceMotion) { el.textContent = target; return; }
    const t0 = performance.now(), dur = 1000;
    const step = (now) => {
      const t = Math.min(1, (now - t0) / dur);
      el.textContent = Math.round(target * (1 - Math.pow(1 - t, 3)));
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      e.target.classList.add("in");
      $$("[data-count]", e.target).forEach(countUp);
      $$("[data-w]", e.target).forEach((b) => { b.style.width = b.dataset.w + "%"; });
      io.unobserve(e.target);
    });
  }, { threshold: 0.12 });
  $$(".reveal").forEach((el) => io.observe(el));

  /* ---------- SLD: fault / trip simulation ---------- */
  (function sld() {
    const svg = $("#sld");
    const path = $("#goosePath"), pkt = $("#goosePacket"), hb = $("#heartbeat");
    const readout = $("#sldReadout"), live = $("#sldLive"), brk = $("#brkState");
    const faultBtn = $("#faultBtn"), resetBtn = $("#resetBtn");
    const len = path.getTotalLength();
    const TRIP_MS = 9, TRAVEL = 900; // shown ms, animation ms (~100x slower)
    let state = "normal", timers = [];

    const say = (txt, cls = "") => { readout.textContent = txt; readout.className = "sld-readout " + cls; };
    const place = (c, t) => { const p = path.getPointAtLength(len * t); c.setAttribute("cx", p.x); c.setAttribute("cy", p.y); };
    function travel(c, dur, onStep, done) {
      const t0 = performance.now();
      let raf;
      const f = (now) => {
        const t = Math.min(1, (now - t0) / dur);
        place(c, t);
        onStep && onStep(t);
        if (t < 1) raf = requestAnimationFrame(f); else done && done();
      };
      raf = requestAnimationFrame(f);
      return () => cancelAnimationFrame(raf);
    }

    let stopHb = null;
    function heartbeat() {
      if (state !== "normal" || reduceMotion || document.hidden) return;
      hb.style.opacity = 0.9;
      stopHb = travel(hb, 1600, null, () => { hb.style.opacity = 0; });
    }
    setInterval(heartbeat, 2600);
    heartbeat();

    faultBtn.addEventListener("click", () => {
      if (state !== "normal") return;
      state = "fault";
      stopHb && stopHb();
      hb.style.opacity = 0;
      faultBtn.disabled = true;
      svg.classList.add("faulted");
      live.textContent = "● FAULT";
      live.classList.add("trip");
      say("FAULT DETECTED · publisher relay picks up…", "warn");

      const finish = () => {
        pkt.style.opacity = 0;
        svg.classList.add("tripped");
        brk.textContent = "OPEN";
        live.textContent = "● TRIPPED";
        state = "tripped";
        say("TRIP ✓ · breaker 52 open · sub-10 ms relay-to-relay (lab-measured, ±5 ms)", "ok");
        resetBtn.disabled = false;
      };
      if (reduceMotion) { finish(); return; }
      timers.push(setTimeout(() => {
        pkt.style.opacity = 1;
        travel(pkt, TRAVEL, (t) => say(`GOOSE trip message in flight · ${(t * TRIP_MS).toFixed(1)} ms`, "warn"), finish);
      }, 450));
    });

    resetBtn.addEventListener("click", () => {
      timers.forEach(clearTimeout);
      timers = [];
      svg.classList.remove("faulted", "tripped");
      brk.textContent = "CLOSED";
      live.textContent = "● LIVE GOOSE";
      live.classList.remove("trip");
      say("NORMAL · relays healthy · GOOSE heartbeat on the wire");
      resetBtn.disabled = true;
      faultBtn.disabled = false;
      state = "normal";
      faultBtn.focus();
    });
  })();
  /* ---------- HERO TYPEWRITER ---------- */
  (function typer() {
    const el = $("#typer");
    if (!el) return;
    const words = ["protection & control.", "PLC automation.", "IEC 61850 substations.", "engineering software.", "the road to automotive design."];
    if (reduceMotion) { el.textContent = words[0]; return; }
    let w = 0, i = 0, del = false;
    const tick = () => {
      const word = words[w];
      i += del ? -1 : 1;
      el.textContent = word.slice(0, i);
      let wait = del ? 28 : 55;
      if (!del && i === word.length) { del = true; wait = 1600; }
      else if (del && i === 0) { del = false; w = (w + 1) % words.length; wait = 300; }
      setTimeout(tick, wait);
    };
    tick();
  })();

  /* ---------- CARD SPOTLIGHT (follows the cursor) ---------- */
  if (matchMedia("(hover: hover)").matches && !reduceMotion) {
    document.addEventListener("pointermove", (e) => {
      const card = e.target.closest(".spot");
      if (!card) return;
      const r = card.getBoundingClientRect();
      card.style.setProperty("--mx", `${e.clientX - r.left}px`);
      card.style.setProperty("--my", `${e.clientY - r.top}px`);
    }, { passive: true });
  }

  /* ---------- COMMAND PALETTE (Ctrl/⌘+K or /) ---------- */
  (function palette() {
    const dlg = $("#palette"), input = $("#paletteInput"), list = $("#paletteList");
    const go = (sel) => () => goTo(sel);
    const cmds = [
      ...$$('#navLinks a[href^="#"]').map((a) => ({ label: "Go to " + a.textContent, hint: "section", run: go(a.getAttribute("href")) })),
      { label: "Inject a fault in the substation diagram", hint: "demo", run: () => { goTo("#top"); setTimeout(() => $("#faultBtn").click(), 500); } },
      { label: "Play the Analyzer Toolkit build log", hint: "demo", run: () => { const t = $('.sw-tab[data-tab="log"]'); goTo(".sw-card"); t && t.click(); } },
      { label: "Toggle light / dark theme", hint: "T", run: () => themeBtn.click() },
      { label: "Open resume", hint: "R", run: () => { location.href = "resume.html"; } },
      { label: "Copy email address", hint: "action", run: () => $("#copyEmail").click() },
      { label: "Request early access to Analyzer Toolkit", hint: "action", run: () => { location.href = `mailto:${P.email}?subject=${encodeURIComponent("Analyzer Toolkit — early access")}`; } },
      ...LOG.map((e) => ({ label: e.title, hint: TYPE_LABEL[e.type].toLowerCase(), run: () => openLog(e.id) })),
      ...allSkills.map((s) => ({ label: s.name, hint: "skill", run: () => activateSkill(s.name) }))
    ];
    let shown = [], sel = 0;
    const render = () => {
      const q = input.value.trim().toLowerCase();
      shown = cmds.filter((c) => !q || c.label.toLowerCase().includes(q) || c.hint.includes(q)).slice(0, 9);
      sel = Math.min(sel, Math.max(0, shown.length - 1));
      list.innerHTML = shown.length
        ? shown.map((c, i) => `<li role="option" id="pal-${i}" aria-selected="${i === sel}" data-i="${i}"><span>${esc(c.label)}</span><kbd>${esc(c.hint)}</kbd></li>`).join("")
        : '<li class="pal-empty">No matches</li>';
      input.setAttribute("aria-activedescendant", shown.length ? "pal-" + sel : "");
    };
    const open = () => { input.value = ""; sel = 0; render(); dlg.showModal(); input.focus(); };
    const run = (i) => { const c = shown[i]; if (!c) return; dlg.close(); setTimeout(c.run, 50); };
    input.addEventListener("input", () => { sel = 0; render(); });
    input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") { e.preventDefault(); sel = (sel + 1) % Math.max(1, shown.length); render(); }
      else if (e.key === "ArrowUp") { e.preventDefault(); sel = (sel - 1 + shown.length) % Math.max(1, shown.length); render(); }
      else if (e.key === "Enter") { e.preventDefault(); run(sel); }
    });
    list.addEventListener("click", (e) => { const li = e.target.closest("[data-i]"); if (li) run(+li.dataset.i); });
    dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
    $("#paletteBtn").addEventListener("click", open);

    document.addEventListener("keydown", (e) => {
      const typing = /input|textarea|select/i.test(document.activeElement.tagName) || dlg.open;
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); dlg.open ? dlg.close() : open(); return; }
      if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "/") { e.preventDefault(); open(); }
      else if (e.key.toLowerCase() === "t") themeBtn.click();
      else if (e.key.toLowerCase() === "r") location.href = "resume.html";
    });
  })();
})();
