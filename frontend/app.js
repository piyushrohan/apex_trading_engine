(function () {
  const { useEffect, useMemo, useRef, useState } = React;

  function num(value, digits = 4) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "-";
    }
    return Number(value).toFixed(digits);
  }

  function TerminalApp() {
    const [status, setStatus] = useState(null);
    const [explain, setExplain] = useState(null);
    const [paperMetrics, setPaperMetrics] = useState(null);
    const [portfolio, setPortfolio] = useState(null);
    const [lastError, setLastError] = useState("");
    const wsRef = useRef(null);

    const apiBase = useMemo(() => {
      const fromQuery = new URLSearchParams(window.location.search).get("api");
      return fromQuery || "http://localhost:8080";
    }, []);

    useEffect(() => {
      let mounted = true;

      async function hydrate() {
        try {
          const [s, e, p, m] = await Promise.all([
            fetch(`${apiBase}/status`).then((r) => r.json()),
            fetch(`${apiBase}/explain/latest`).then((r) => r.json()).catch(() => null),
            fetch(`${apiBase}/portfolio`).then((r) => r.json()),
            fetch(`${apiBase}/metrics/paper`).then((r) => r.json()),
          ]);
          if (!mounted) return;
          setStatus(s);
          setExplain(e);
          setPortfolio(p);
          setPaperMetrics(m);
          setLastError("");
        } catch (err) {
          if (mounted) setLastError(String(err));
        }
      }

      function connectWs() {
        const wsUrl = apiBase.replace("http://", "ws://").replace("https://", "wss://");
        const ws = new WebSocket(`${wsUrl}/ws/status`);
        wsRef.current = ws;

        ws.onmessage = (evt) => {
          try {
            const msg = JSON.parse(evt.data);
            setStatus(msg);
            if (msg.last_explanation) setExplain(msg.last_explanation);
          } catch (err) {
            setLastError(String(err));
          }
        };
        ws.onerror = () => {
          setLastError("WebSocket disconnected; fallback polling active.");
        };
      }

      hydrate();
      connectWs();
      const pollId = window.setInterval(hydrate, 5000);
      return () => {
        mounted = false;
        window.clearInterval(pollId);
        if (wsRef.current) wsRef.current.close();
      };
    }, [apiBase]);

    const mode = status?.operator_mode || "paper";
    const hedge = explain?.hedge || {};
    const candidates = hedge.candidates || {};
    const candidateNames = Object.keys(candidates);

    return React.createElement(
      "main",
      { className: "terminal" },
      React.createElement(
        "section",
        { className: "banner" },
        React.createElement(
          "div",
          null,
          React.createElement("strong", null, "APEX Terminal"),
          React.createElement(
            "div",
            { className: "k" },
            `${status?.symbol || "ETHUSDC"} | model ${status?.model_id || "-"}`
          )
        ),
        React.createElement(
          "div",
          { className: `mode-pill ${mode === "live" ? "mode-live" : "mode-paper"}` },
          mode.toUpperCase()
        )
      ),
      React.createElement(
        "section",
        { className: "grid" },
        React.createElement(
          "article",
          { className: "panel" },
          React.createElement("h2", null, "Market + Risk"),
          row("Regime", status?.regime || "-"),
          row("Mark", num(status?.mark_price, 2)),
          row(
            "Kill Switch",
            React.createElement(
              "span",
              { className: status?.kill_switch_active ? "status-alert" : "status-ok" },
              status?.kill_switch_active ? "ACTIVE" : "SAFE"
            )
          ),
          row("Book Role", portfolio?.runtime?.role || "-"),
          row("Long Qty", num(portfolio?.runtime?.long_qty, 4)),
          row("Short Qty", num(portfolio?.runtime?.short_qty, 4)),
          row("Equity", num(portfolio?.runtime?.equity, 2))
        ),
        React.createElement(
          "article",
          { className: "panel" },
          React.createElement("h2", null, "Paper Performance"),
          row("Snapshots", paperMetrics?.snapshots ?? "-"),
          row("Sharpe", num(paperMetrics?.sharpe, 4)),
          row("Max DD", num(paperMetrics?.max_drawdown, 4)),
          row("Final Equity", num(paperMetrics?.final_equity, 2)),
          row("Directional Signals", paperMetrics?.directional_decisions ?? "-")
        ),
        React.createElement(
          "article",
          { className: "panel" },
          React.createElement("h2", null, "Explainability"),
          row("Decision", explain?.decision || "-"),
          row("Conviction", num(explain?.conviction_score, 4)),
          row("Summary", explain?.summary || "-"),
          React.createElement(
            "div",
            { className: "k", style: { marginTop: "8px" } },
            "Primary Reasons"
          ),
          React.createElement(
            "ul",
            { className: "reason-list" },
            ...(explain?.primary_reasons || []).map((reason, idx) =>
              React.createElement("li", { key: idx }, reason)
            )
          )
        ),
        React.createElement(
          "article",
          { className: "panel" },
          React.createElement("h2", null, "Hedge Selector"),
          row("Selection Mode", hedge.selection_mode || "-"),
          row("Selected", hedge.selected || "NONE"),
          row("Bandit Arm", hedge.bandit_arm || "-"),
          row("Exploration", String(Boolean(hedge.exploration))),
          React.createElement(
            "div",
            { className: "k", style: { marginTop: "8px" } },
            "Candidate Scores"
          ),
          React.createElement(
            "div",
            { className: "candidates" },
            ...(candidateNames.length
              ? candidateNames.map((name) =>
                  React.createElement(
                    "div",
                    { className: "candidate-row", key: name },
                    React.createElement("span", null, name),
                    React.createElement("span", null, num(candidates[name], 4))
                  )
                )
              : [React.createElement("span", { key: "none" }, "-")])
          )
        )
      ),
      lastError
        ? React.createElement("section", { className: "panel status-alert" }, lastError)
        : null
    );
  }

  function row(key, value) {
    return React.createElement(
      "div",
      { className: "kv", key },
      React.createElement("div", { className: "k" }, key),
      React.createElement("div", null, value)
    );
  }

  const root = ReactDOM.createRoot(document.getElementById("root"));
  root.render(React.createElement(TerminalApp));
})();
