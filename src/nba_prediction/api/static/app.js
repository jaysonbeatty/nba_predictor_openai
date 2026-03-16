const gameDateInput = { value: todayLocalIso() };
const resultsSection = document.querySelector("#resultsSection");
const baselineCards = document.querySelector("#baselineCards");
const scenarioResults = document.querySelector("#scenarioResults");
const baselineTitle = document.querySelector("#baselineTitle");
const statusBanner = document.querySelector("#statusBanner");
const chatThread = document.querySelector("#chatThread");
const chatForm = document.querySelector("#chatForm");
const chatInput = document.querySelector("#chatInput");
const sendBtn = document.querySelector("#sendBtn");
const starterPills = [...document.querySelectorAll(".pill")];
const gamesList = document.querySelector("#gamesList");
const gamesTitle = document.querySelector("#gamesTitle");

let todaysGames = [];

const TEAM_ALIASES = {
  ATL: ["atl", "atlanta", "hawks"],
  BOS: ["bos", "boston", "celtics"],
  BKN: ["bkn", "brooklyn", "nets"],
  CHA: ["cha", "charlotte", "hornets"],
  CHI: ["chi", "chicago", "bulls"],
  CLE: ["cle", "cleveland", "cavs", "cavaliers"],
  DAL: ["dal", "dallas", "mavericks", "mavs"],
  DEN: ["den", "denver", "nuggets"],
  DET: ["det", "detroit", "pistons"],
  GSW: ["gsw", "golden state", "warriors"],
  HOU: ["hou", "houston", "rockets"],
  IND: ["ind", "indiana", "pacers"],
  LAC: ["lac", "clippers", "la clippers"],
  LAL: ["lal", "lakers", "la lakers"],
  MEM: ["mem", "memphis", "grizzlies"],
  MIA: ["mia", "miami", "heat"],
  MIL: ["mil", "milwaukee", "bucks"],
  MIN: ["min", "minnesota", "timberwolves", "wolves"],
  NOP: ["nop", "new orleans", "pelicans", "pels"],
  NYK: ["nyk", "new york", "knicks"],
  OKC: ["okc", "oklahoma city", "thunder"],
  ORL: ["orl", "orlando", "magic"],
  PHI: ["phi", "philadelphia", "sixers", "76ers"],
  PHX: ["phx", "phoenix", "suns"],
  POR: ["por", "portland", "trail blazers", "blazers"],
  SAC: ["sac", "sacramento", "kings"],
  SAS: ["sas", "san antonio", "spurs"],
  TOR: ["tor", "toronto", "raptors"],
  UTA: ["uta", "utah", "jazz"],
  WAS: ["was", "washington", "wizards"],
};

function todayLocalIso() {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function normalizeText(text) {
  return text.toLowerCase().replace(/[^\w\s]/g, " ").replace(/\s+/g, " ").trim();
}

function titleCaseName(name) {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function showStatus(message, isError = false) {
  statusBanner.textContent = message;
  statusBanner.classList.remove("hidden");
  statusBanner.style.background = isError ? "rgba(197, 81, 47, 0.14)" : "rgba(36, 92, 69, 0.12)";
  statusBanner.style.color = isError ? "#8f2e1b" : "#245c45";
}

function clearStatus() {
  statusBanner.classList.add("hidden");
}

function addChatMessage(role, html) {
  const article = document.createElement("article");
  article.className = `chat-message ${role}`;
  article.innerHTML = html;
  chatThread.appendChild(article);
  chatThread.scrollTop = chatThread.scrollHeight;
}

function setSendingState(isSending) {
  sendBtn.disabled = isSending;
  sendBtn.textContent = isSending ? "Thinking..." : "Send";
}

function formatGameLabel(game) {
  return `${game.away_team} at ${game.home_team}`;
}

function renderGamesList() {
  gamesTitle.textContent = todaysGames.length
    ? `${todaysGames.length} game${todaysGames.length === 1 ? "" : "s"} on ${gameDateInput.value}`
    : `No games on ${gameDateInput.value}`;

  gamesList.innerHTML = "";
  if (!todaysGames.length) {
    gamesList.innerHTML = `<p class="empty-games">No games available for ${gameDateInput.value}.</p>`;
    return;
  }

  todaysGames.forEach((game) => {
    const card = document.createElement("article");
    card.className = "game-chip";
    card.innerHTML = `<strong>${game.away_team}</strong><span>at</span><strong>${game.home_team}</strong>`;
    gamesList.appendChild(card);
  });
}

async function loadGames() {
  clearStatus();
  const response = await fetch(`/games?date=${encodeURIComponent(gameDateInput.value)}`);
  const data = await response.json();
  todaysGames = data.games || [];
  renderGamesList();
  if (todaysGames.length) {
    showStatus(`Loaded ${todaysGames.length} game${todaysGames.length === 1 ? "" : "s"} for ${gameDateInput.value}.`);
  } else {
    showStatus(`No games found for ${gameDateInput.value}.`, true);
  }
}

function renderMetricCard(label, value, caption) {
  return `
    <article class="metric-card">
      <h3>${label}</h3>
      <span class="metric-value">${value}</span>
      <p>${caption}</p>
    </article>
  `;
}

function renderBaseline(result) {
  baselineTitle.textContent = `${result.game.away_team} at ${result.game.home_team}`;
  const baseline = result.baseline_prediction;
  baselineCards.innerHTML = [
    renderMetricCard("Home Win Probability", `${(baseline.home_win_prob * 100).toFixed(1)}%`, result.game.home_team),
    renderMetricCard("Away Win Probability", `${(baseline.away_win_prob * 100).toFixed(1)}%`, result.game.away_team),
    renderMetricCard("Projected Margin", `${baseline.predicted_margin.toFixed(2)}`, `${result.game.home_team} minus ${result.game.away_team}`),
    renderMetricCard("Projected Total", `${baseline.predicted_total.toFixed(2)}`, "Combined points"),
  ].join("");
}

function renderScenarioCard(scenario) {
  const prediction = scenario.scenario_prediction;
  const delta = scenario.delta;
  const resolved = scenario.resolved_overrides || [];
  const resolvedText = resolved.length
    ? resolved
        .map((item) => `${item.player_name} (${item.resolved_team}) - ${item.status}${item.minutes_limit ? ` ${item.minutes_limit} min` : ""}`)
        .join(", ")
    : "No overrides applied.";

  return `
    <article class="scenario-result-card">
      <h3>${scenario.name}</h3>
      <p>Home win probability: <strong>${(prediction.home_win_prob * 100).toFixed(1)}%</strong></p>
      <p>Projected margin: <strong>${prediction.predicted_margin.toFixed(2)}</strong></p>
      <p>Projected total: <strong>${prediction.predicted_total.toFixed(2)}</strong></p>
      <h4>Delta vs baseline</h4>
      <p>Win probability change: <strong>${(delta.home_win_prob_change * 100).toFixed(1)} pts</strong></p>
      <p>Margin change: <strong>${delta.predicted_margin_change.toFixed(2)}</strong></p>
      <p>Total change: <strong>${delta.predicted_total_change.toFixed(2)}</strong></p>
      <div class="resolved-list"><strong>Resolved overrides:</strong> ${resolvedText}</div>
    </article>
  `;
}

function renderResults(data) {
  renderBaseline(data);
  scenarioResults.innerHTML = data.scenarios.map(renderScenarioCard).join("");
  resultsSection.classList.remove("hidden");
}

function parseMinutesLimit(text) {
  const match = text.match(/(?:limited|limit(?:ed)? to)\s+(\d{1,2})/i);
  return match ? Number(match[1]) : null;
}

function splitScenarioParts(text) {
  const compareMatch = text.match(/compare\s+(.+)/i);
  if (!compareMatch) return null;

  let content = compareMatch[1].trim();
  content = content.replace(/^baseline\s*,?\s*/i, "");
  const parts = content
    .split(/\s*(?:\band\b|,)\s*/i)
    .map((part) => part.trim())
    .filter(Boolean);

  return parts.length ? parts : null;
}

function inferStatus(part) {
  if (/\blimited\b/i.test(part)) return "limited";
  if (/\bavailable\b|\bback\b|\breturns?\b/i.test(part)) return "available";
  return "out";
}

function inferPlayerName(part) {
  let cleaned = part
    .replace(/^what happens if\s+/i, "")
    .replace(/^what if\s+/i, "")
    .replace(/^compare\s+/i, "")
    .replace(/\btonight\b/gi, "")
    .replace(/\btoday\b/gi, "")
    .replace(/\bbaseline\b/gi, "")
    .replace(/\bfor\b\s+[a-z\s]+$/i, "")
    .replace(/\bif\b/gi, "")
    .replace(/\bis\b|\bare\b/gi, "")
    .replace(/\bout\b|\blimited\b|\bavailable\b/gi, "")
    .replace(/\bto\b\s+\d{1,2}\b/gi, "")
    .replace(/\bminutes?\b/gi, "")
    .replace(/[?.!]/g, "")
    .trim();

  return titleCaseName(cleaned.replace(/\s+/g, " "));
}

function chooseBestPlayerMatch(matches) {
  if (!matches.length) return null;
  const [best, second] = matches;
  if (!second) return best;
  if ((best.match_score || 0) >= 0.9) return best;
  if ((best.match_score || 0) - (second.match_score || 0) >= 0.08) return best;
  if ((best.player_name || "").toLowerCase() === (second.player_name || "").toLowerCase()) return best;
  return null;
}

function buildSingleScenarioFromPrompt(prompt) {
  const playerName = inferPlayerName(prompt);
  if (!playerName) return null;
  const status = inferStatus(prompt);
  const minutesLimit = status === "limited" ? parseMinutesLimit(prompt) : null;
  const label = status === "limited" && minutesLimit
    ? `${playerName} limited to ${minutesLimit}`
    : `${playerName} ${status}`;
  const override = { player_name: playerName, status };
  if (minutesLimit) override.minutes_limit = minutesLimit;
  return { name: label, overrides: [override] };
}

function buildComparePayload(prompt) {
  const parts = splitScenarioParts(prompt);
  if (!parts) return null;
  const scenarios = parts.map((part) => buildSingleScenarioFromPrompt(part)).filter(Boolean);
  return scenarios.length ? scenarios : null;
}

async function inferGameFromPrompt(prompt) {
  const normalized = normalizeText(prompt);
  const matches = todaysGames.filter((game) => {
    const homeAliases = TEAM_ALIASES[game.home_team] || [];
    const awayAliases = TEAM_ALIASES[game.away_team] || [];
    return [...homeAliases, ...awayAliases].some((alias) => normalized.includes(alias));
  });

  if (matches.length === 1) return matches[0];
  if (matches.length > 1) return { ambiguous: true, matches };

  const inferredPlayer = inferPlayerName(prompt);
  if (inferredPlayer) {
    const response = await fetch(`/players?query=${encodeURIComponent(inferredPlayer)}`);
    if (response.ok) {
      const data = await response.json();
      const teamMatches = (data.matches || []).filter((match) =>
        todaysGames.some((game) => game.home_team === match.team || game.away_team === match.team)
      );

      const bestTeamMatch = chooseBestPlayerMatch(teamMatches);
      if (bestTeamMatch) {
        const resolvedTeam = bestTeamMatch.team;
        const game = todaysGames.find((item) => item.home_team === resolvedTeam || item.away_team === resolvedTeam);
        if (game) return game;
      }

      if (teamMatches.length === 1) {
        const resolvedTeam = teamMatches[0].team;
        const game = todaysGames.find((item) => item.home_team === resolvedTeam || item.away_team === resolvedTeam);
        if (game) return game;
      }

      if (teamMatches.length > 1) {
        const possibleGames = todaysGames.filter((game) =>
          teamMatches.some((match) => game.home_team === match.team || game.away_team === match.team)
        );
        return { ambiguous: true, matches: possibleGames };
      }
    }
  }

  if (todaysGames.length === 1) return todaysGames[0];
  return null;
}

function isOutlookPrompt(prompt) {
  return /\boutlook\b|\bwho wins\b|\bforecast\b|\bchance\b|\bprobability\b/i.test(prompt) && !/\bout\b|\blimited\b|\bcompare\b/i.test(prompt);
}

async function runBaselineForGame(game) {
  const response = await fetch("/simulate-game", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      date: gameDateInput.value,
      home_team: game.home_team,
      away_team: game.away_team,
      overrides: [],
    }),
  });
  return response.json().then((data) => ({ ok: response.ok, data }));
}

function formatOutlookReply(data) {
  const baseline = data.baseline_prediction;
  return `
    <p><strong>${data.game.away_team} at ${data.game.home_team}</strong></p>
    <p>${data.game.home_team} win probability: <strong>${(baseline.home_win_prob * 100).toFixed(1)}%</strong>. ${data.game.away_team} win probability: <strong>${(baseline.away_win_prob * 100).toFixed(1)}%</strong>.</p>
    <p>Projected score: <strong>${baseline.expected_home_score}-${baseline.expected_away_score}</strong>, with a projected margin of <strong>${baseline.predicted_margin.toFixed(2)}</strong> and total of <strong>${baseline.predicted_total.toFixed(2)}</strong>.</p>
  `;
}

function formatAssistantReply(data) {
  const baseline = data.baseline_prediction;
  const scenario = data.scenarios[0]?.scenario_prediction;
  const delta = data.scenarios[0]?.delta;
  const resolved = data.scenarios[0]?.resolved_overrides || [];

  if (!scenario || !delta) {
    return `<p>I ran the comparison, but I couldn’t format the scenario response cleanly.</p>`;
  }

  const assumption = resolved.length
    ? `<p class="assumption">Assuming you mean ${resolved.map((item) => `${item.player_name} (${item.resolved_team})`).join(", ")}.</p>`
    : "";

  return `
    ${assumption}
    <p><strong>Baseline:</strong> ${data.game.away_team} ${Math.round(baseline.away_win_prob * 100)}%, projected score ${baseline.expected_home_score}-${baseline.expected_away_score}.</p>
    <p><strong>Scenario:</strong> ${data.game.away_team} ${Math.round(scenario.away_win_prob * 100)}%, projected score ${scenario.expected_home_score}-${scenario.expected_away_score}.</p>
    <p><strong>Change:</strong> ${data.game.away_team} win probability ${delta.away_win_prob_change >= 0 ? "+" : ""}${(delta.away_win_prob_change * 100).toFixed(1)} pts, margin ${delta.predicted_margin_change >= 0 ? "+" : ""}${delta.predicted_margin_change.toFixed(2)}, total ${delta.predicted_total_change >= 0 ? "+" : ""}${delta.predicted_total_change.toFixed(2)}.</p>
  `;
}

async function runChatPrompt(prompt) {
  clearStatus();
  if (!todaysGames.length) {
    showStatus("No games are loaded for tonight yet.", true);
    return;
  }

  const inferred = await inferGameFromPrompt(prompt);
  if (!inferred) {
    const labels = todaysGames.map(formatGameLabel).join(", ");
    addChatMessage("assistant", `<p>I couldn't tell which game you meant. Tonight's games are: ${labels}.</p>`);
    return;
  }
  if (inferred.ambiguous) {
    addChatMessage("assistant", `<p>I found more than one possible game. Please mention one of these matchups: ${inferred.matches.map(formatGameLabel).join(", ")}.</p>`);
    return;
  }

  if (isOutlookPrompt(prompt)) {
    const { ok, data } = await runBaselineForGame(inferred);
    if (!ok) {
      showStatus(data.detail || "I couldn't fetch that outlook.", true);
      addChatMessage("assistant", `<p>${data.detail || "I couldn't fetch that outlook."}</p>`);
      return;
    }
    renderBaseline(data);
    scenarioResults.innerHTML = "";
    resultsSection.classList.remove("hidden");
    addChatMessage("assistant", formatOutlookReply(data));
    showStatus(`Loaded outlook for ${formatGameLabel(inferred)}.`);
    return;
  }

  const compareScenarios = buildComparePayload(prompt);
  const fallbackScenario = buildSingleScenarioFromPrompt(prompt);
  const scenarios = compareScenarios || (fallbackScenario ? [fallbackScenario] : []);

  if (!scenarios.length) {
    addChatMessage(
      "assistant",
      "<p>I can help with prompts like <strong>“What is the outlook for Atlanta tonight?”</strong>, <strong>“What happens if Jalen Brunson is out tonight?”</strong>, or <strong>“Compare Brunson out and Towns out for New York tonight.”</strong></p>"
    );
    return;
  }

  const response = await fetch("/compare-scenarios", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      date: gameDateInput.value,
      home_team: inferred.home_team,
      away_team: inferred.away_team,
      scenarios,
    }),
  });

  const data = await response.json();
  if (!response.ok) {
    showStatus(data.detail || "The comparison request failed.", true);
    addChatMessage("assistant", `<p>${data.detail || "I couldn't run that scenario."}</p>`);
    return;
  }

  renderResults(data);
  addChatMessage("assistant", formatAssistantReply(data));
  showStatus(`Compared ${data.scenarios.length} scenario${data.scenarios.length === 1 ? "" : "s"} for ${formatGameLabel(inferred)}.`);
}

async function onSubmit(event) {
  event.preventDefault();
  const prompt = chatInput.value.trim();
  if (!prompt) return;

  addChatMessage("user", `<p>${prompt}</p>`);
  chatInput.value = "";
  setSendingState(true);
  try {
    await runChatPrompt(prompt);
  } finally {
    setSendingState(false);
  }
}

chatForm.addEventListener("submit", onSubmit);
starterPills.forEach((pill) => {
  pill.addEventListener("click", () => {
    chatInput.value = pill.dataset.prompt || "";
    chatInput.focus();
  });
});

addChatMessage(
  "assistant",
  "<p>I’m focused on tonight’s slate. Ask something like <strong>“What is the outlook for Atlanta tonight?”</strong> or <strong>“What happens if Jalen Brunson is out tonight?”</strong></p>"
);

loadGames();
