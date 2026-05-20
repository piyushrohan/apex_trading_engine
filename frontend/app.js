(function () {
  const { useEffect, useMemo, useRef, useState } = React;
  const h = React.createElement;

  const TABS = [
    "Live",
    "Paper",
    "Explain",
    "Hedge",
    "Models",
    "History",
    "Risk",
    "Logs",
  ];

  function num(value, digits = 4) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "-";
    }
    return Number(value).toFixed(digits);
  }

  function pct(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "-";
    }
    return `${(Number(value) * 100).toFixed(digits)}%`;
  }

  function ago(timestamp) {
    if (!timestamp) return "never";
    const then = new Date(timestamp).getTime();
    if (!Number.isFinite(then)) return "unknown";
    const seconds = Math.max(0, Math.round((Date.now() - then) / 1000));
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    return `${Math.floor(minutes / 60)}h ago`;
  }

  async function getJson(url, fallback = null) {
    const response = await fetch(url);
    if (!response.ok) return fallback;
    return response.json();
  }

  async function postJson(url, body) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || response.statusText);
    }
    return payload;
  }

  function TerminalApp() {
    const [activeTab, setActiveTab] = useState("Live");
    const [autoRefresh, setAutoRefresh] = useState(true);
    const [decisionFilter, setDecisionFilter] = useState("");
    const [reason, setReason] = useState("operator review");
    const [modeRequest, setModeRequest] = useState("paper");
    const [riskProfile, setRiskProfile] = useState("balanced");
    const [state, setState] = useState({
      status: null,
      explain: null,
      paper: null,
      portfolio: null,
      hedge: null,
      gate: null,
      decisions: { items: [], total: 0 },
      equity: { items: [], total: 0 },
      market: { ohlcv: [], market: [] },
      models: { models: {} },
      promotion: null,
      logs: { files: [] },
      audit: { items: [], total: 0 },
      controls: null,
      error: "",
      refreshedAt: null,
      wsState: "connecting",
    });
    const wsRef = useRef(null);

    const apiBase = useMemo(() => {
      const fromQuery = new URLSearchParams(window.location.search).get("api");
      return fromQuery || "http://localhost:8080";
    }, []);

    async function hydrate() {
      const decisionQuery = decisionFilter
        ? `&decision=${encodeURIComponent(decisionFilter)}`
        : "";
      try {
        const [
          status,
          explain,
          portfolio,
          paper,
          hedge,
          gate,
          decisions,
          equity,
          market,
          models,
          promotion,
          logs,
          audit,
          controls,
        ] = await Promise.all([
          getJson(`${apiBase}/status`, null),
          getJson(`${apiBase}/explain/latest`, null),
          getJson(`${apiBase}/portfolio`, null),
          getJson(`${apiBase}/metrics/paper`, null),
          getJson(`${apiBase}/reports/hedge`, null),
          getJson(`${apiBase}/live/gate`, null),
          getJson(`${apiBase}/history/decisions?limit=80${decisionQuery}`, {
            items: [],
            total: 0,
          }),
          getJson(`${apiBase}/history/equity?limit=500`, { items: [], total: 0 }),
          getJson(`${apiBase}/history/market?limit=160`, { ohlcv: [], market: [] }),
          getJson(`${apiBase}/models`, { models: {} }),
          getJson(`${apiBase}/models/promotion/status`, null),
          getJson(`${apiBase}/logs/runtime?limit=80`, { files: [] }),
          getJson(`${apiBase}/audit?limit=80`, { items: [], total: 0 }),
          getJson(`${apiBase}/control/state`, null),
        ]);
        setState((current) => ({
          ...current,
          status,
          explain,
          portfolio,
          paper,
          hedge,
          gate,
          decisions,
          equity,
          market,
          models,
          promotion,
          logs,
          audit,
          controls,
          error: "",
          refreshedAt: new Date().toISOString(),
        }));
      } catch (err) {
        setState((current) => ({ ...current, error: String(err) }));
      }
    }

    async function sendCommand(command, extra = {}) {
      try {
        const result = await postJson(`${apiBase}/control/${command}`, {
          confirm: true,
          reason,
          ...extra,
        });
        setState((current) => ({
          ...current,
          controls: result.state_after || current.controls,
          audit: {
            ...current.audit,
            items: [result, ...(current.audit?.items || [])],
            total: (current.audit?.total || 0) + 1,
          },
          error: "",
        }));
        await hydrate();
      } catch (err) {
        setState((current) => ({ ...current, error: String(err) }));
      }
    }

    useEffect(() => {
      hydrate();
    }, [apiBase, decisionFilter]);

    useEffect(() => {
      if (!autoRefresh) return undefined;
      const pollId = window.setInterval(hydrate, 5000);
      return () => window.clearInterval(pollId);
    }, [autoRefresh, apiBase, decisionFilter]);

    useEffect(() => {
      const wsUrl = apiBase.replace("http://", "ws://").replace("https://", "wss://");
      const ws = new WebSocket(`${wsUrl}/ws/status`);
      wsRef.current = ws;
      ws.onopen = () => setState((current) => ({ ...current, wsState: "live" }));
      ws.onclose = () => setState((current) => ({ ...current, wsState: "offline" }));
      ws.onerror = () => setState((current) => ({ ...current, wsState: "error" }));
      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          setState((current) => ({
            ...current,
            status: msg,
            explain: msg.last_explanation || current.explain,
            refreshedAt: new Date().toISOString(),
          }));
        } catch (err) {
          setState((current) => ({ ...current, error: String(err) }));
        }
      };
      return () => ws.close();
    }, [apiBase]);

    return h(
      "main",
      { className: "terminal" },
      h(Header, {
        apiBase,
        autoRefresh,
        state,
        onRefresh: hydrate,
        onToggleRefresh: () => setAutoRefresh((value) => !value),
      }),
      h(
        "nav",
        { className: "tabs", "aria-label": "Terminal sections" },
        ...TABS.map((tab) =>
          h(
            "button",
            {
              key: tab,
              className: activeTab === tab ? "tab tab-active" : "tab",
              onClick: () => setActiveTab(tab),
            },
            tab
          )
        )
      ),
      state.error ? h("section", { className: "alert-line" }, state.error) : null,
      activeTab === "Live"
        ? h(LiveView, {
            state,
            reason,
            setReason,
            modeRequest,
            setModeRequest,
            riskProfile,
            setRiskProfile,
            sendCommand,
          })
        : null,
      activeTab === "Paper" ? h(PaperView, { state }) : null,
      activeTab === "Explain" ? h(ExplainView, { explain: state.explain }) : null,
      activeTab === "Hedge" ? h(HedgeView, { explain: state.explain, hedge: state.hedge }) : null,
      activeTab === "Models" ? h(ModelsView, { state }) : null,
      activeTab === "History"
        ? h(HistoryView, {
            state,
            decisionFilter,
            setDecisionFilter,
            onRefresh: hydrate,
          })
        : null,
      activeTab === "Risk" ? h(RiskView, { state }) : null,
      activeTab === "Logs" ? h(LogsView, { state }) : null
    );
  }

  function Header({ apiBase, autoRefresh, state, onRefresh, onToggleRefresh }) {
    const status = state.status || {};
    const mode = status.operator_mode || "paper";
    const stale = status.updated_at
      ? Date.now() - new Date(status.updated_at).getTime() > 15000
      : true;
    return h(
      "section",
      { className: "hero" },
      h(
        "div",
        { className: "hero-main" },
        h("div", { className: "eyebrow" }, "APEX Operator Cockpit"),
        h("h1", null, status.symbol || "ETHUSDC"),
        h(
          "div",
          { className: "hero-sub" },
          `model ${status.model_id || "-"} | ${apiBase}`
        )
      ),
      h(
        "div",
        { className: "hero-actions" },
        h(StatusPill, {
          label: state.wsState === "live" ? "WS live" : `WS ${state.wsState}`,
          tone: state.wsState === "live" ? "ok" : "warn",
        }),
        h(StatusPill, { label: mode.toUpperCase(), tone: mode === "live" ? "warn" : "ok" }),
        h(StatusPill, { label: stale ? "STALE" : `updated ${ago(status.updated_at)}`, tone: stale ? "danger" : "ok" }),
        h("button", { className: "icon-button", onClick: onRefresh, title: "Refresh" }, "Refresh"),
        h(
          "button",
          { className: "icon-button", onClick: onToggleRefresh, title: "Toggle auto refresh" },
          autoRefresh ? "Auto on" : "Auto off"
        )
      )
    );
  }

  function LiveView({
    state,
    reason,
    setReason,
    modeRequest,
    setModeRequest,
    riskProfile,
    setRiskProfile,
    sendCommand,
  }) {
    const status = state.status || {};
    const portfolio = state.portfolio?.runtime || status.portfolio || {};
    const controls = state.controls || {};
    return h(
      "section",
      { className: "layout two-one" },
      h(
        "div",
        { className: "stack" },
        h(Panel, { title: "Market + Risk" },
          h(KpiGrid, {
            items: [
              ["Regime", status.regime || "-"],
              ["Mark", num(status.mark_price, 2)],
              ["Kill Switch", status.kill_switch_active ? "ACTIVE" : "SAFE"],
              ["Ingestion", status.ingestion_enabled ? "ON" : "OFF"],
              ["Hedge", status.hedge_enabled ? "ON" : "OFF"],
              ["Updated", ago(status.updated_at)],
            ],
          })
        ),
        h(Panel, { title: "Position Book" },
          h(KpiGrid, {
            items: [
              ["Role", portfolio.role || "-"],
              ["Long Qty", num(portfolio.long_qty, 4)],
              ["Short Qty", num(portfolio.short_qty, 4)],
              ["Net Qty", num(portfolio.net_qty, 4)],
              ["Gross Qty", num(portfolio.gross_qty, 4)],
              ["Equity", num(portfolio.equity, 2)],
            ],
          })
        )
      ),
      h(
        Panel,
        { title: "Control Deck", accent: "warn" },
        h(
          "p",
          { className: "muted" },
          "Commands are recorded as auditable operator intent for the runtime pipeline."
        ),
        h("label", { className: "field-label" }, "Command reason"),
        h("input", {
          className: "field",
          value: reason,
          onChange: (event) => setReason(event.target.value),
        }),
        h(
          "div",
          { className: "button-grid" },
          h("button", { onClick: () => sendCommand("pause") }, "Pause"),
          h("button", { onClick: () => sendCommand("resume") }, "Resume"),
          h("button", { className: "danger-button", onClick: () => sendCommand("kill-switch") }, "Kill"),
          h("button", { onClick: () => sendCommand("clear-kill-switch") }, "Clear kill"),
          h("button", { className: "danger-button", onClick: () => sendCommand("flatten") }, "Flatten"),
        ),
        h("label", { className: "field-label" }, "Mode request"),
        h(
          "div",
          { className: "inline-controls" },
          h(
            "select",
            { className: "field", value: modeRequest, onChange: (e) => setModeRequest(e.target.value) },
            h("option", { value: "paper" }, "paper"),
            h("option", { value: "shadow" }, "shadow"),
            h("option", { value: "live" }, "live")
          ),
          h("button", { onClick: () => sendCommand("set-mode", { mode: modeRequest }) }, "Request mode")
        ),
        h("label", { className: "field-label" }, "Risk profile request"),
        h(
          "div",
          { className: "inline-controls" },
          h("input", {
            className: "field",
            value: riskProfile,
            onChange: (e) => setRiskProfile(e.target.value),
          }),
          h("button", { onClick: () => sendCommand("set-risk-profile", { profile: riskProfile }) }, "Request risk")
        ),
        h("pre", { className: "json-box" }, JSON.stringify(controls, null, 2))
      )
    );
  }

  function PaperView({ state }) {
    const metrics = state.paper || {};
    const equity = state.equity?.items || [];
    return h(
      "section",
      { className: "layout one-one" },
      h(Panel, { title: "Paper Performance" },
        h(KpiGrid, {
          items: [
            ["Snapshots", metrics.snapshots ?? "-"],
            ["Sharpe", num(metrics.sharpe, 4)],
            ["Max DD", pct(metrics.max_drawdown, 2)],
            ["Final Equity", num(metrics.final_equity, 2)],
            ["Directional", metrics.directional_decisions ?? "-"],
            ["Fill Rate", pct(metrics.fill_rate, 2)],
          ],
        }),
        h(Sparkline, { rows: equity, valueKey: "equity", label: "Equity history" })
      ),
      h(Panel, { title: "Equity Snapshots" },
        h(Table, {
          rows: equity.slice(-12).reverse(),
          columns: ["timestamp", "equity", "long_qty", "short_qty", "regime"],
        })
      )
    );
  }

  function ExplainView({ explain }) {
    const buckets = explain?.confidence_buckets || {};
    return h(
      "section",
      { className: "layout two-one" },
      h(
        "div",
        { className: "stack" },
        h(Panel, { title: "Decision Decode" },
          h(KpiGrid, {
            items: [
              ["Decision", explain?.decision || "-"],
              ["Conviction", num(explain?.conviction_score, 4)],
              ["Tier", explain?.summary?.confidence_tier || "-"],
              ["Model", explain?.executing_model || "-"],
              ["Regime", explain?.active_regime || "-"],
              ["Model ID", explain?.model_id || "-"],
            ],
          }),
          h(ProbabilityBars, { values: explain?.raw_action_probs || [] })
        ),
        h(Panel, { title: "Confidence Buckets" },
          h(BarList, {
            rows: Object.keys(buckets).map((name) => ({
              name,
              value: Math.abs(Number(buckets[name].score || 0)),
              detail: `${buckets[name].alignment || "-"} | ${num(buckets[name].score, 4)}`,
            })),
          })
        )
      ),
      h(Panel, { title: "Narrative + Lifecycle" },
        h(ListBlock, { title: "Primary Reasons", rows: explain?.primary_reasons || [] }),
        h(ListBlock, { title: "Risk Factors", rows: explain?.risk_factors || [] }),
        h(ListBlock, { title: "Market Narrative", rows: explain?.market_narrative || [] }),
        h("pre", { className: "json-box" }, JSON.stringify(explain?.position_lifecycle || {}, null, 2))
      )
    );
  }

  function HedgeView({ explain, hedge }) {
    const current = explain?.hedge || {};
    const candidates = current.candidates || {};
    const strategies = hedge?.strategies || {};
    return h(
      "section",
      { className: "layout one-one" },
      h(Panel, { title: "Current Hedge Selection" },
        h(KpiGrid, {
          items: [
            ["Mode", current.selection_mode || "-"],
            ["Selected", current.selected || "NONE"],
            ["Selected Score", num(current.selected_score, 4)],
            ["Bandit Arm", current.bandit_arm || "-"],
            ["Exploration", String(Boolean(current.exploration))],
            ["Intent", current.proposal?.intent || "-"],
          ],
        }),
        h(BarList, {
          rows: Object.keys(candidates).map((name) => ({
            name,
            value: Number(candidates[name] || 0),
            detail: num(candidates[name], 4),
          })),
        })
      ),
      h(Panel, { title: "Hedge Attribution" },
        h(KpiGrid, {
          items: [
            ["Window Days", hedge?.window_days ?? "-"],
            ["Total Selected", hedge?.total_selected ?? "-"],
          ],
        }),
        h(Table, {
          rows: Object.keys(strategies).map((name) => ({ strategy: name, ...strategies[name] })),
          columns: ["strategy", "selected_count", "score_observations", "avg_score", "pnl"],
        })
      )
    );
  }

  function ModelsView({ state }) {
    const models = state.models?.models || {};
    const rows = Object.keys(models).map((id) => ({ model_id: id, ...models[id] }));
    const promotion = state.promotion || {};
    return h(
      "section",
      { className: "layout one-one" },
      h(Panel, { title: "Registry" },
        h(KpiGrid, {
          items: [
            ["Active Prod", state.models?.active_prod || "-"],
            ["Active Shadow", state.models?.active_shadow || "-"],
            ["Registered", rows.length],
          ],
        }),
        h(Table, { rows, columns: ["model_id", "type", "status", "created_at"] })
      ),
      h(Panel, { title: "Promotion Readiness" },
        h(KpiGrid, {
          items: [
            ["Shadow", promotion.active_shadow || "-"],
            ["Decision", promotion.decision?.action || "-"],
            ["Reason", promotion.decision?.reason || "-"],
            ["Shadow Sharpe", num(promotion.shadow_metrics?.sharpe, 4)],
            ["Primary Sharpe", num(promotion.primary_metrics?.sharpe, 4)],
            ["Min Trades", promotion.thresholds?.min_shadow_trades ?? "-"],
          ],
        }),
        h("pre", { className: "json-box" }, JSON.stringify(promotion, null, 2))
      )
    );
  }

  function HistoryView({ state, decisionFilter, setDecisionFilter, onRefresh }) {
    const decisions = state.decisions?.items || [];
    const market = state.market?.ohlcv || [];
    return h(
      "section",
      { className: "stack" },
      h(Panel, { title: "Decision Journal" },
        h(
          "div",
          { className: "toolbar" },
          h("label", { className: "field-label" }, "Decision"),
          h(
            "select",
            { className: "field small-field", value: decisionFilter, onChange: (e) => setDecisionFilter(e.target.value) },
            h("option", { value: "" }, "all"),
            h("option", { value: "LONG" }, "long"),
            h("option", { value: "SHORT" }, "short"),
            h("option", { value: "FLAT" }, "flat")
          ),
          h("button", { onClick: onRefresh }, "Reload")
        ),
        h(Table, {
          rows: decisions,
          columns: ["timestamp", "decision", "conviction_score", "active_regime", "model_id"],
        })
      ),
      h(Panel, { title: "Market Replay Slice" },
        h(Sparkline, { rows: market, valueKey: "close", label: "Close price" }),
        h(Table, {
          rows: market.slice(-18).reverse(),
          columns: ["timestamp", "open", "high", "low", "close", "volume"],
        })
      )
    );
  }

  function RiskView({ state }) {
    const gate = state.gate || {};
    const status = state.status || {};
    return h(
      "section",
      { className: "layout one-one" },
      h(Panel, { title: "Paper To Live Gate", accent: gate.passed ? "ok" : "danger" },
        h(KpiGrid, {
          items: [
            ["Gate", gate.passed ? "PASSED" : "BLOCKED"],
            ["Live Enabled", String(Boolean(gate.live_enabled))],
            ["Skip Gate", String(Boolean(gate.skip_paper_gate))],
            ["Paper Days", num(gate.metrics?.paper_days, 2)],
            ["Min Days", gate.metrics?.min_days ?? "-"],
            ["Min Trades", gate.metrics?.min_trades ?? "-"],
            ["Sharpe", num(gate.metrics?.sharpe, 4)],
            ["Max DD", pct(gate.metrics?.max_drawdown, 2)],
          ],
        }),
        h(ListBlock, { title: "Blocking Reasons", rows: gate.reasons || [] })
      ),
      h(Panel, { title: "Runtime Safety" },
        h(KpiGrid, {
          items: [
            ["Kill Switch", status.kill_switch_active ? "ACTIVE" : "SAFE"],
            ["Operator Mode", status.operator_mode || "-"],
            ["Hedge Enabled", String(Boolean(status.hedge_enabled))],
            ["Ingestion", String(Boolean(status.ingestion_enabled))],
            ["Last Update", ago(status.updated_at)],
          ],
        }),
        h("pre", { className: "json-box" }, JSON.stringify(state.controls || {}, null, 2))
      )
    );
  }

  function LogsView({ state }) {
    const logFiles = state.logs?.files || [];
    const audit = state.audit?.items || [];
    return h(
      "section",
      { className: "layout one-one" },
      h(Panel, { title: "Runtime Logs" },
        logFiles.length
          ? logFiles.map((file) =>
              h("div", { className: "log-file", key: file.path },
                h("div", { className: "muted" }, file.path),
                h("pre", { className: "log-box" }, (file.lines || []).join("\n"))
              )
            )
          : h("p", { className: "muted" }, "No log files found in logs/.")
      ),
      h(Panel, { title: "Audit Trail" },
        h(Table, { rows: audit, columns: ["timestamp", "command", "reason"] })
      )
    );
  }

  function Panel({ title, children, accent }) {
    const className = accent ? `panel panel-${accent}` : "panel";
    return h("article", { className }, h("h2", null, title), children);
  }

  function StatusPill({ label, tone }) {
    return h("span", { className: `pill pill-${tone || "neutral"}` }, label);
  }

  function KpiGrid({ items }) {
    return h(
      "div",
      { className: "kpi-grid" },
      ...items.map(([label, value]) =>
        h("div", { className: "kpi", key: label }, h("span", null, label), h("strong", null, value))
      )
    );
  }

  function BarList({ rows }) {
    const max = Math.max(0.0001, ...rows.map((row) => Math.abs(Number(row.value || 0))));
    return h(
      "div",
      { className: "bars" },
      ...(rows.length
        ? rows.map((row) =>
            h("div", { className: "bar-row", key: row.name },
              h("div", { className: "bar-label" }, row.name),
              h("div", { className: "bar-track" },
                h("div", {
                  className: "bar-fill",
                  style: { width: `${Math.min(100, (Math.abs(row.value) / max) * 100)}%` },
                })
              ),
              h("div", { className: "bar-value" }, row.detail)
            )
          )
        : [h("div", { className: "muted", key: "empty" }, "No scores yet.")])
    );
  }

  function ProbabilityBars({ values }) {
    const labels = ["LONG", "FLAT", "SHORT"];
    return h(BarList, {
      rows: values.map((value, idx) => ({
        name: labels[idx] || `A${idx}`,
        value: Number(value || 0),
        detail: pct(value, 2),
      })),
    });
  }

  function Sparkline({ rows, valueKey, label }) {
    const values = rows
      .map((row) => Number(row[valueKey]))
      .filter((value) => Number.isFinite(value));
    if (values.length < 2) {
      return h("div", { className: "empty-chart" }, `${label}: not enough history yet`);
    }
    const width = 540;
    const height = 140;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    const points = values
      .map((value, index) => {
        const x = (index / Math.max(1, values.length - 1)) * width;
        const y = height - ((value - min) / span) * (height - 12) - 6;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
    return h(
      "div",
      { className: "chart-wrap" },
      h("div", { className: "chart-label" }, `${label} (${num(min, 2)} - ${num(max, 2)})`),
      h(
        "svg",
        { className: "sparkline", viewBox: `0 0 ${width} ${height}`, role: "img" },
        h("polyline", { points, fill: "none", strokeWidth: "3" })
      )
    );
  }

  function ListBlock({ title, rows }) {
    return h(
      "div",
      { className: "list-block" },
      h("h3", null, title),
      rows && rows.length
        ? h("ul", null, ...rows.map((row, idx) => h("li", { key: idx }, String(row))))
        : h("p", { className: "muted" }, "None")
    );
  }

  function Table({ rows, columns }) {
    if (!rows || !rows.length) {
      return h("p", { className: "muted" }, "No rows yet.");
    }
    return h(
      "div",
      { className: "table-scroll" },
      h(
        "table",
        null,
        h("thead", null, h("tr", null, ...columns.map((column) => h("th", { key: column }, column)))),
        h(
          "tbody",
          null,
          ...rows.map((row, idx) =>
            h(
              "tr",
              { key: idx },
              ...columns.map((column) =>
                h("td", { key: column }, formatCell(row[column]))
              )
            )
          )
        )
      )
    );
  }

  function formatCell(value) {
    if (value === null || value === undefined) return "-";
    if (typeof value === "number") return Math.abs(value) >= 100 ? num(value, 2) : num(value, 4);
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  }

  const root = ReactDOM.createRoot(document.getElementById("root"));
  root.render(h(TerminalApp));
})();
