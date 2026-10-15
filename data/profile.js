/* ==========================================================================
   PROFILE DATA — the single source of truth for the website AND resume.html.

   To track something new, add an entry to the right list below and commit.
   No build step: GitHub Pages serves this file as-is.

   Formatting:
   - Dates are "YYYY" or "YYYY-MM". Use end: "present" for ongoing items.
     Leave `end` out for one-off items (a single date).
   - Wrap text in **double asterisks** to make it bold.
   - `skills` on log/software entries must match a skill `name` exactly —
     that is what links skills <-> experience on the site.
   - `resume: false` hides an entry from resume.html but keeps it on the site.
   - `site: false` keeps an entry on resume.html only (the site is
     engineering-focused; everything else lives on LinkedIn).
   ========================================================================== */

window.PROFILE = {
  updated: "2026-10",

  name: "Haider Rizvi",
  title: "Electrical Engineering Technologist",
  designation: "Technologist-in-Training (T.T.), ASET",
  location: "Calgary, Alberta, Canada",
  email: "hasanhaider2001@gmail.com",
  phone: "+1 (587) 973-1140",
  phoneHref: "+15879731140",
  linkedin: "https://linkedin.com/in/haiderizvi",
  github: "https://github.com/haider1243-tech",
  startYear: 2021, // first year in engineering — drives the "years" stat

  summary:
    "**SAIT Electrical Engineering Technology Graduate (2026)** and ASET Technologist-in-Training with practical lab and capstone project experience. Proficient in **Rockwell PLC/HMI programming**, industrial networking, and **SEL protective relay** configuration. Skilled in **AutoCAD drafting** and applying **CEC criteria** for wire and conduit sizing. Combines technical troubleshooting with standard field safety practices (**LOTO, FLHA**), and builds browser-based engineering software for process analyzer systems. Seeking an entry-level electrical technologist, automation/controls, testing, field service, or design support role.",
  seeking: "Entry-level electrical technologist, automation/controls, testing, field service, or design support roles",

  /* ---------------- SKILLS ----------------
     color: "blueprint" | "amber" | "cyan"
     icon:  plc | screen | clock | network | bolt | target | wave | chart | draw | cube | code
     learning: true  -> shown dashed with a "learning" tag */
  skills: [
    {
      group: "Automation & Controls", color: "blueprint",
      items: [
        { name: "Studio 5000", icon: "plc", desc: "Programmed Allen-Bradley PLCs for lab automation projects" },
        { name: "RSLogix 5000", icon: "plc", desc: "Ladder logic development and I/O mapping" },
        { name: "FactoryTalk View", icon: "screen", desc: "Custom HMI screen configuration for operator control" },
        { name: "RSLinx Enterprise", icon: "clock", desc: "Industrial network communication between PLCs and devices" },
        { name: "Unity Pro XL", icon: "plc", desc: "Modicon PLC programming for automation logic" },
        { name: "RSNetWorx", icon: "network", desc: "Industrial network configuration and diagnostics" },
        { name: "PowerSuite", icon: "clock", desc: "Programmed ATV 71 VFDs and ATS48 soft starters" }
      ]
    },
    {
      group: "Power & Protection", color: "amber",
      items: [
        { name: "SEL AcSELerator QuickSet", icon: "bolt", desc: "Configured SEL-751A relay settings and protection logic" },
        { name: "SEL AcSELerator Architect", icon: "bolt", desc: "Built IEC 61850 GOOSE publisher/subscriber messaging" },
        { name: "ETAP", icon: "plc", desc: "Simulated and tested industrial protection schemes" },
        { name: "Wireshark", icon: "target", desc: "Verified peer-to-peer relay trip operation on the network" },
        { name: "ASE2000 (DNP3)", icon: "plc", desc: "Simulated SCADA traffic and verified DNP3 protocol comms" },
        { name: "IEC 61850", icon: "wave", desc: "Core protocol used in capstone GOOSE messaging design" },
        { name: "CEC Codebook", icon: "chart", desc: "Applied Sections 4 & 12 (Tables 1-9 & 13-50) for ampacity, temperature derating, and conduit fill" }
      ]
    },
    {
      group: "Equipment & Lab", color: "amber",
      items: [
        { name: "Allen-Bradley PLCs", icon: "plc", desc: "ControlLogix programming — ladder logic interlocks, I/O mapping, fault handling" },
        { name: "PanelView 5310 HMI", icon: "screen", desc: "Ran custom FactoryTalk View HMI screens on physical PanelView hardware" },
        { name: "SEL-751A / 587 / 501 relays", icon: "bolt", desc: "Configured and tested industrial protection schemes on real relays" },
        { name: "ATV 71 VFD / ATS48 soft starter", icon: "wave", desc: "Programmed drives and soft starters via PowerSuite" },
        { name: "LabVolt / LVDC-EMS", icon: "chart", desc: "Wired, commissioned, and tested transformers, DC/induction, and synchronous machines" }
      ]
    },
    {
      group: "Design & Drafting", color: "cyan",
      items: [
        { name: "AutoCAD Electrical", icon: "draw", desc: "Drafted single-line diagrams (SLDs), schematics, panel board layouts, and capstone wiring diagrams" },
        { name: "Autodesk Revit", icon: "draw", desc: "Actively learning — building on AutoCAD fundamentals for BIM-based design", learning: true },
        { name: "SolidWorks", icon: "cube", desc: "3D modeling coursework — where the automotive design interest started" },
        { name: "ProjectLibre", icon: "network", desc: "Project scheduling and deadline tracking for capstone work" }
      ]
    },
    {
      group: "Software & Engineering Tools", color: "blueprint",
      items: [
        { name: "HTML / CSS / JavaScript", icon: "code", desc: "Single-file, no-install browser tools that run on PC, tablet, and phone" },
        { name: "DXF / CAD export", icon: "draw", desc: "Layered, full-size DXF drawings and SolidWorks macros generated from the browser" },
        { name: "three.js", icon: "cube", desc: "Interactive true-size 3D model of an analyzer sample tap" },
        { name: "Automated testing", icon: "target", desc: "Headless backtest suite — hand-calculated cases, random simulations, layout checks" }
      ]
    }
  ],

  /* ---------------- SOFTWARE PROJECTS ----------------
     Products and tools you're building.
     launched: false -> shown as "In development · not yet public", with no link
                        and a "Request early access" button instead.
     When it launches: set launched: true and add url: "https://..." */
  software: [
    {
      id: "analyzer-toolkit",
      name: "Analyzer Toolkit",
      tagline: "Browser-based engineering checks for process analyzer systems.",
      version: "v3.4",
      status: "active",
      start: "2026-10",
      launched: false,
      availability: "In development · commercial release planned",
      note: "Not publicly available yet — the toolkit and its product website are still being built for a paid release.",
      desc: "A single HTML file — no install, no server — that checks where a sample probe tip lands in the pipe bore from the valve / nipple / threadolet stack, and suggests fittings or probe lengths that fix it.",
      features: [
        "Probe insertion check against the centre third (or a custom window), with a live dimensioned section drawing",
        "Tolerance stack-up: worst case, RSS, and a seeded **10,000-build Monte Carlo**",
        "Wake frequency screening (fs/fn ≤ 0.4, ASME PTC 19.3 TW) and velocity profile across the bore",
        "Exports: 11x17 PDF sheet, layered **DXF** for AutoCAD, **SolidWorks** macro + equations, probe schedule CSV/PDF",
        "Interactive true-size **3D model** of the tap (three.js)",
        "**102-check** headless backtest, incl. a 3,000-tap random simulation against an independent formula"
      ],
      skills: ["HTML / CSS / JavaScript", "DXF / CAD export", "three.js", "Automated testing", "SolidWorks", "AutoCAD Electrical"],
      roadmap: [
        { text: "Probe insertion check", state: "done" },
        { text: "Analyzer config checker", state: "todo" },
        { text: "Wake frequency tool", state: "todo" },
        { text: "Sample lag time tool", state: "todo" },
        { text: "Product website", state: "active" },
        { text: "Public (paid) launch", state: "todo" }
      ],
      // Build log — played as a terminal on the site. Newest last.
      releases: [
        { v: "v1.0.0", date: "2026-10-04", note: "Probe insertion check, live section drawing, saved taps" },
        { v: "v2.0.0", date: "2026-10-05", note: "Redline markups, revision history, title block, DXF/SVG/CSV export" },
        { v: "v3.0.0", date: "2026-10-05", note: "Probe length mode, custom target zone" },
        { v: "v3.1.0", date: "2026-10-05", note: "Job-agnostic defaults, load example" },
        { v: "v3.2.0", date: "2026-10-15", note: "CAD package: DXF + SolidWorks macro + equations" },
        { v: "v3.3.0", date: "2026-10-15", note: "11x17 vector PDF sheet, SolidWorks drawing → DWG" },
        { v: "v3.4.0", date: "2026-10-15", note: "3D model, Monte Carlo stack-up, wake frequency, 102-check backtest" }
      ]
    }
  ],

  /* ---------------- PATHWAY LOGBOOK ----------------
     type: "work" | "education" | "project" | "milestone" */
  log: [
    {
      id: "analyzer-toolkit-log", type: "project",
      start: "2026-10", end: "present",
      title: "Analyzer Toolkit — v1.0 → v3.4",
      org: "Software product · in development",
      summary: "Built and shipped a browser-based probe insertion checker for process analyzer systems — seven releases in under two weeks.",
      details: [
        "Released v1.0 (probe insertion check + live section drawing) on Oct 4, 2026 and iterated to **v3.4** by Oct 15.",
        "Added CAD exports (DXF, SolidWorks macro, 11x17 PDF), a 3D model, tolerance stack-up, and wake frequency screening.",
        "Wrote a **102-check** headless backtest covering math, layouts, every export, and offline behaviour."
      ],
      skills: ["HTML / CSS / JavaScript", "DXF / CAD export", "three.js", "Automated testing", "SolidWorks"],
      resume: true
    },
    {
      id: "capstone", type: "project",
      start: "2026-04",
      title: "Capstone Project Lead — IEC 61850 Digital Substation Control & Protection",
      org: "SAIT · Capstone Design",
      summary: "Led an IEC 61850 substation protection scheme modeled around utility protection concepts observed during an ENMAX field visit — relay config, wiring, and verified trip testing.",
      details: [
        "Led design, **relay configuration**, **AutoCAD wiring diagrams**, testing, and technical reporting for an **IEC 61850-based substation protection scheme** modeled around utility protection concepts observed during an ENMAX field visit.",
        "Configured **SEL-751A protection relays** for **GOOSE publisher/subscriber messaging** using **SEL AcSELerator Architect** and **QuickSet**, replacing hardwired trip signaling with Ethernet-based relay communication.",
        "Verified peer-to-peer relay trip operation using **Wireshark** and **QuickSet**, observing **sub-10 ms lab response times** with documented ±5 ms timestamp uncertainty.",
        "Delivered as a proof-of-concept using available lab equipment — with an honest understanding that fiber-based hardwired schemes remain preferred in many production environments today."
      ],
      skills: ["SEL AcSELerator QuickSet", "SEL AcSELerator Architect", "Wireshark", "IEC 61850", "SEL-751A / 587 / 501 relays", "AutoCAD Electrical", "ProjectLibre"],
      resume: true
    },
    {
      id: "sait", type: "education",
      start: "2024-09", end: "2026-04",
      title: "Electrical Engineering Technology Diploma",
      org: "Southern Alberta Institute of Technology (SAIT)", location: "Calgary, AB",
      summary: "Switched deliberately into EET. Lab and project work across automation, protection, machines, and electrical design.",
      coursework: "**PLC ControlLogix**, Industrial Networks, **Protection and Control**, Power System Analysis, Electrical Machines, Instrumentation, **AutoCAD Electrical**, Capstone Design",
      experienceTitle: "SAIT — Electrical Engineering Technology Lab & Project Work",
      details: [
        "**Industrial Automation:** Programmed **Allen-Bradley** and **Modicon PLCs** using **Studio 5000, RSLogix 5000,** and **Unity Pro XL**. Mapped I/O, developed ladder logic interlocks, and configured custom HMI screens via **FactoryTalk View Studio** to run physical PanelView 5310 hardware.",
        "**Power Systems & Protection:** Configured and tested industrial protection schemes using **SEL-751A/587/501 relays** and **ETAP** software. Used **ASE2000** to simulate SCADA traffic and verify **DNP3 protocol** communication.",
        "**Electrical Drafting & CEC Sizing:** Drafted **AutoCAD single-line diagrams (SLDs)**, schematics, and panel board layouts. Applied **CEC Sections 4 and 12 (Tables 1-9 & 13-50)** to calculate precise wire ampacities, temperature derating, and conduit fill.",
        "**Electrical Machines & Drives:** Wired, commissioned, and tested three-phase transformers, DC/induction motors, and synchronous machines using **LabVolt/LVDC-EMS**. Programmed **ATV 71 VFDs** and **ATS48 Soft Starters** via PowerSuite.",
        "**Field Safety Compliance:** Completed **daily FLHA forms** and executed strict **LOTO procedures** before wiring control panels, switches, sensors, and terminal blocks."
      ],
      skills: ["Studio 5000", "RSLogix 5000", "Unity Pro XL", "FactoryTalk View", "RSLinx Enterprise", "RSNetWorx", "PowerSuite", "Allen-Bradley PLCs", "PanelView 5310 HMI", "SEL-751A / 587 / 501 relays", "ATV 71 VFD / ATS48 soft starter", "LabVolt / LVDC-EMS", "ETAP", "ASE2000 (DNP3)", "AutoCAD Electrical", "CEC Codebook"],
      resume: true
    },
    {
      id: "allisons", type: "work",
      start: "2024-04", end: "present",
      title: "Construction / Renovation Assistant",
      org: "Allisons Construction Ltd.", location: "Calgary, AB · Casual / On-Call",
      summary: "Residential renovation work including drywall, cabinets, flooring/tile, painting, lumber cutting, drilling/fastening, and material handling, with safe hand/power tool use across job sites.",
      details: [],
      skills: [],
      resume: true,
      site: false // resume only — non-engineering work lives on LinkedIn
    },
    {
      id: "uofc", type: "education",
      start: "2021", end: "2024",
      title: "Mechanical Engineering Coursework",
      org: "University of Calgary — Schulich School of Engineering", location: "Calgary, AB",
      summary: "Started in mechanical engineering. Found SolidWorks — genuinely enjoyed the 3D modeling side of the craft.",
      details: [],
      skills: ["SolidWorks"],
      resume: true
    }
  ],

  /* ---------------- CHALLENGES & LESSONS ----------------
     Problems you've hit and how you solved them.
     area: "software" | "engineering" | "career" */
  challenges: [
    {
      date: "2026-10", area: "software", project: "Analyzer Toolkit",
      title: "Bad inputs still produced a drawing",
      problem: "Negative, zero, or impossible inputs (like a nipple too short to make up) still gave a result and a broken drawing.",
      fix: "Added input validation that blocks the result and every export until the inputs are fixed.",
      lesson: "An engineering tool should refuse to answer rather than give a confident wrong answer."
    },
    {
      date: "2026-10", area: "software", project: "Analyzer Toolkit",
      title: "Exports only worked in one environment",
      problem: "Export buttons were hidden when the tool was opened outside the environment it was built in.",
      fix: "Reworked exports to download normally in any browser, and added a headless test that runs every export in a plain browser.",
      lesson: "Test where the user will actually run it, not just where you built it."
    },
    {
      date: "2026-10", area: "software", project: "Analyzer Toolkit",
      title: "Phone layout scrolled sideways",
      problem: "Wide tables stretched the page to about 2× the screen width, and sections overlapped on small Android screens.",
      fix: "Fixed the layout, enlarged tap targets, and pinned the live result bar to the bottom of the screen. Layouts are now checked on iPhone, Android, tablet, and desktop in the test suite.",
      lesson: "Field tools get used on phones — design for the smallest screen first."
    },
    {
      date: "2026-10", area: "software", project: "Analyzer Toolkit",
      title: "Closed CAD formats",
      problem: "A browser can't write .dwg or native SolidWorks files directly — they're closed formats.",
      fix: "Generated a layered DXF plus a SolidWorks macro that has SolidWorks build the parts, assembly, and drawing and export the DWG itself.",
      lesson: "When you can't produce the format, produce the instructions for the tool that can."
    },
    {
      date: "2026-04", area: "engineering", project: "Capstone",
      title: "Lab equipment vs. production practice",
      problem: "The capstone had to prove Ethernet-based GOOSE tripping using the lab equipment available, while fiber-based hardwired schemes remain preferred in many production substations.",
      fix: "Scoped it as a proof-of-concept and verified the trip on the wire with Wireshark — sub-10 ms lab response, with the ±5 ms timestamp uncertainty documented rather than hidden.",
      lesson: "Be clear about what a result proves — and what it doesn't."
    },
    {
      date: "2024", area: "career", project: "Pathway",
      title: "Changing direction mid-degree",
      problem: "Three years into mechanical engineering, the focus wasn't where I wanted to build a career.",
      fix: "Switched deliberately into Electrical Engineering Technology at SAIT — and kept what stuck: SolidWorks and 3D modeling.",
      lesson: "A switch isn't starting over if you carry the useful parts forward."
    }
  ],

  /* ---------------- GOALS TRACKER ----------------
     status: "done" | "active" | "planned"
     milestone state: "done" | "active" | "todo"  (progress = done / total) */
  goals: [
    {
      phase: "00", when: "2024 – 2026", title: "Technologist Foundation", area: "Electrical", status: "done",
      desc: "Diploma, professional registration, and a verified capstone.",
      milestones: [
        { text: "EET Diploma — SAIT", state: "done" },
        { text: "ASET Technologist-in-Training registration", state: "done" },
        { text: "Capstone: verified GOOSE trip (sub-10 ms)", state: "done" }
      ]
    },
    {
      phase: "00", when: "Now", title: "Ship Analyzer Toolkit", area: "Software", status: "active",
      desc: "Grow the toolkit from one tool into a full set of analyzer system checks — then launch it as a paid product with its own website.",
      milestones: [
        { text: "Probe insertion check (v3.4)", state: "done" },
        { text: "Analyzer config checker", state: "todo" },
        { text: "Wake frequency tool", state: "todo" },
        { text: "Sample lag time tool", state: "todo" },
        { text: "Product website", state: "active" },
        { text: "Public (paid) launch", state: "todo" }
      ]
    },
    {
      phase: "00", when: "Now", title: "Add Revit to the AutoCAD foundation", area: "Design", status: "active",
      desc: "BIM-based design on top of AutoCAD Electrical fundamentals.",
      milestones: [
        { text: "AutoCAD Electrical fundamentals", state: "done" },
        { text: "Revit fundamentals", state: "active" },
        { text: "First complete Revit model", state: "todo" }
      ]
    },
    {
      phase: "01", when: "Year one", title: "Vehicle Electrical Architecture", area: "Electrical", status: "planned",
      desc: "Mapping wiring harnesses, fusing, and low-voltage distribution — applying protection and control fundamentals to a smaller-scale, higher-stakes system.",
      milestones: [
        { text: "Wiring harness & fusing fundamentals", state: "todo" },
        { text: "Low-voltage distribution", state: "todo" }
      ]
    },
    {
      phase: "02", when: "Year two", title: "EV Powertrain Control", area: "Electrical", status: "planned",
      desc: "Battery management systems and motor control logic — a direct extension of relay and PLC thinking into automotive-scale systems.",
      milestones: [
        { text: "Battery management systems", state: "todo" },
        { text: "Motor control logic", state: "todo" }
      ]
    },
    {
      phase: "03", when: "Ongoing", title: "Chassis & Component Design", area: "Mechanical", status: "planned",
      desc: "Back to SolidWorks — the software that started it all — modeling automotive components against real structural and mechanical constraints.",
      milestones: [
        { text: "SolidWorks refresh", state: "todo" },
        { text: "Model a component against real constraints", state: "todo" }
      ]
    },
    {
      phase: "04", when: "Year three+", title: "Sensor & ADAS Integration", area: "Electrical + Mechanical", status: "planned",
      desc: "Protection-relay logic — detect, decide, act — applied to driver-assist sensor systems once the fundamentals above are locked in.",
      milestones: [
        { text: "Sensor fundamentals", state: "todo" },
        { text: "Detect → decide → act logic for ADAS", state: "todo" }
      ]
    }
  ],

  /* ---------------- CERTIFICATIONS & EDUCATION ----------------
     kind: "cert" counts toward the certifications stat */
  certs: [
    { name: "Technologist-in-Training (T.T.)", issuer: "ASET · Active Registration", kind: "cert", featured: true },
    { name: "Electrical Engineering Technology Diploma", issuer: "SAIT · Calgary, AB · 2026", kind: "education" },
    { name: "Mechanical Engineering Coursework", issuer: "University of Calgary (Schulich) · 2021–2024", kind: "education" },
    { name: "CSTS 2020", issuer: "Safety Certification", kind: "cert" },
    { name: "WHMIS 2015", issuer: "Safety Certification", kind: "cert" },
    { name: "TDG Awareness", issuer: "Safety Certification", kind: "cert" }
  ],

  languages: "English, Urdu / Hindi",
  workStatus: "Open Work Permit (PGWP)"
};
