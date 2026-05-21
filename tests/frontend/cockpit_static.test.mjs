import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const app = await readFile(new URL("../../frontend/app.js", import.meta.url), "utf8");

test("probability bars use model action order", () => {
  assert.match(app, /const ACTION_LABELS = \["SHORT", "FLAT", "LONG"\]/);
  assert.match(app, /selected: labels\[idx\] === decision/);
  assert.doesNotMatch(app, /const labels = \["LONG", "FLAT", "SHORT"\]/);
});

test("operator mode selector excludes shadow", () => {
  assert.match(app, /const OPERATOR_MODES = \["paper", "live"\]/);
  assert.doesNotMatch(app, /h\("option", \{ value: "shadow" \}/);
});

test("dangerous commands require an explicit confirmation modal", () => {
  for (const command of ["kill-switch", "clear-kill-switch", "flatten"]) {
    assert.match(app, new RegExp(`"${command}"`));
  }
  assert.match(app, /function needsConfirmation/);
  assert.match(app, /function ConfirmCommandModal/);
  assert.match(app, /aria-label": "Confirm operator command"/);
});

test("control commands still write confirmed operator intent", () => {
  assert.match(app, /confirm: true/);
  assert.match(app, /postJson\(`\$\{apiBase\}\/control\/\$\{command\}`/);
});

test("models cockpit exposes governance readiness and lifecycle history", () => {
  assert.match(app, /\/models\/lifecycle\?limit=12/);
  assert.match(app, /Production Readiness/);
  assert.match(app, /Experiment Runs/);
  assert.match(app, /CANDIDATE", "EVALUATING", "SHADOW", "APPROVED", "PROD"/);
});

test("ops cockpit exposes trader production readiness", () => {
  assert.match(app, /"Ops"/);
  assert.match(app, /\/ops\/readiness/);
  assert.match(app, /Trader Readiness/);
  assert.match(app, /Guardrail Findings/);
  assert.match(app, /Data Freshness/);
  assert.match(app, /PROD blocked/);
});
