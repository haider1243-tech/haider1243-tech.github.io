# haider1243-tech.github.io

Personal site, career pathway tracker, and resume for **Haider Rizvi** — Electrical Engineering Technologist (ASET T.T.), Calgary.

Live at <https://haider1243-tech.github.io>. Static HTML/CSS/JS, served by GitHub Pages. No build step.

## Files

| File | What it is |
|---|---|
| `data/profile.js` | **All the content.** Skills, logbook, software projects, challenges, goals, certs. Edit this to update the site and the resume together. |
| `index.html` | The portfolio page (hero, capstone narrative, section layout). |
| `resume.html` | Printable resume, generated from `data/profile.js`. Use **Print / Save as PDF**. |
| `assets/style.css` | Styles, incl. light/dark themes. |
| `assets/main.js` | Renders the data and runs the interactive parts (fault simulation, command palette, build-log terminal, filters, skill links, goals). |
| `assets/common.js` | Date formatting and helpers shared by both pages. |

## Updating the tracker

Open `data/profile.js` and add to the right list. The comments at the top of each list explain the fields.

- **New job / course / project** → add to `log` (`type: "work" | "education" | "project" | "milestone"`). Use `end: "present"` while it's ongoing; durations and "Ongoing" badges are calculated automatically.
- **Problem you solved** → add to `challenges` with `problem`, `fix`, and `lesson`.
- **Roadmap progress** → in `roadmap`, flip a milestone's `state` from `"todo"` → `"active"` → `"done"`. Tracks: `credentials`, `oilgas`, `automotive`, `bridge`. `target` (year) decides which phase it shows in; the C.E.T. experience clock fills itself from `clock.from`.
- **New skill** → add to a group in `skills`. List the same name in a log entry's `skills` to link them both ways.
- **New software project** → add to `software`. Leave `launched: false` to show it as in development with no link; set `launched: true` and `url` when it's public.
- Bump `updated` at the top so the footer and resume show the right date.

Preview locally: `python3 -m http.server` in this folder, then open <http://localhost:8000>.

## Related repos

- **analyzer-toolkit** (private) — browser-based engineering checks for process analyzer systems. In development for a paid release, so the site shows it with no link and an early-access button. When it launches, set `launched: true` and add `url` in `data/profile.js`.
