#!/usr/bin/env node
/*
 * whykvc headless smoke test — zero npm dependencies.
 *
 * Launches a local Chrome with the DevTools protocol, opens web/index.html via
 * file://, walks the hash routes below at two viewports, and asserts:
 *   - exactly one `section.view.active`
 *   - `body[data-view]` matches the route's first segment
 *   - no Runtime.exceptionThrown / Log error / console.error
 *   - no horizontal page overflow (scrollWidth <= innerWidth + 1)
 *
 * Requires Node 22+ (global fetch + WebSocket).
 *
 * Usage:
 *   node web/tools/smoke.mjs
 *   CHROME_PATH="/path/to/chrome" node web/tools/smoke.mjs
 */

import { spawn } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { pathToFileURL, fileURLToPath } from "node:url";

const PORT = 9333;
const HOST = "127.0.0.1";
const SETTLE_MS = 350;
const ROUTE_TIMEOUT_MS = 8000;
const PAGE_LOAD_TIMEOUT_MS = 20000;
const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const PAGE_URL = pathToFileURL(path.join(REPO_ROOT, "web", "index.html")).href;

const ROUTES = [
  "#think",
  "#kv",
  "#kv/3",
  "#kv/reflect",
  "#prefix/11",
  "#paged/2",
  "#gpu",
  "#gpu/6",
  "#npu",
  "#cuda",
  "#ecn",
  "#transformer",
  "#anatomy",
  "#train",
  "#net",
  "#nccl",
  "#opt",
  "#calc",
  "#feedback",
];

const VIEWPORTS = [
  { name: "1440x900", width: 1440, height: 900 },
  { name: "760x900", width: 760, height: 900 },
];

const CHROME_CANDIDATES = [
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
  path.join(os.homedir(), "Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
  "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
  "/snap/bin/chromium",
  "/opt/google/chrome/chrome",
  "/opt/google/chrome-beta/chrome",
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
];

const PATH_BINARIES = [
  "google-chrome",
  "google-chrome-stable",
  "chromium",
  "chromium-browser",
  "chrome",
];

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function log(message) {
  process.stdout.write(message + "\n");
}

function fail(message) {
  process.stderr.write(message + "\n");
}

function isExecutable(file) {
  try {
    if (!fs.statSync(file).isFile()) return false;
    if (process.platform !== "win32") fs.accessSync(file, fs.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

function which(bin) {
  const dirs = (process.env.PATH || "").split(path.delimiter);
  for (const dir of dirs) {
    if (!dir) continue;
    const candidate = path.join(dir, bin);
    if (isExecutable(candidate)) return candidate;
  }
  return null;
}

function findChrome() {
  const fromEnv = process.env.CHROME_PATH;
  if (fromEnv) {
    if (!isExecutable(fromEnv)) {
      throw new Error(
        `CHROME_PATH is set but is not an executable Chrome binary: ${fromEnv}`,
      );
    }
    return fromEnv;
  }
  for (const candidate of CHROME_CANDIDATES) {
    if (isExecutable(candidate)) return candidate;
  }
  for (const bin of PATH_BINARIES) {
    const found = which(bin);
    if (found) return found;
  }
  throw new Error(
    "Could not find Chrome. Set CHROME_PATH to a Chrome/Chromium executable, " +
      "or install one of: " + PATH_BINARIES.join(", "),
  );
}

function assertPortFree(port) {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", (error) => {
      if (error.code === "EADDRINUSE") {
        reject(
          new Error(
            `Port ${port} is already in use; the smoke test needs it for Chrome's ` +
              "DevTools endpoint. Stop the other process or free the port.",
          ),
        );
      } else {
        reject(error);
      }
    });
    server.once("listening", () => {
      server.close(() => resolve());
    });
    server.listen(port, HOST);
  });
}

function normalizeWsUrl(raw) {
  const url = new URL(raw);
  if (url.hostname === "localhost" || url.hostname === "::1" || url.hostname === "[::1]") {
    url.hostname = HOST;
  }
  return url.href;
}

async function fetchJson(url, timeoutMs = 1500) {
  const response = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
  if (!response.ok) throw new Error(`HTTP ${response.status} from ${url}`);
  return response.json();
}

class CDP {
  constructor(socket) {
    this.socket = socket;
    this.nextId = 1;
    this.pending = new Map();
    this.listeners = new Map();
    this.closed = false;

    socket.addEventListener("message", (event) => {
      let message;
      try {
        message = JSON.parse(String(event.data));
      } catch {
        return;
      }
      if (typeof message.id === "number") {
        const pending = this.pending.get(message.id);
        if (!pending) return;
        this.pending.delete(message.id);
        if (message.error) {
          pending.reject(
            new Error(`${pending.method}: ${message.error.message || JSON.stringify(message.error)}`),
          );
        } else {
          pending.resolve(message.result || {});
        }
        return;
      }
      if (typeof message.method !== "string") return;
      const listeners = this.listeners.get(message.method);
      if (!listeners) return;
      for (const listener of Array.from(listeners)) {
        try {
          listener(message.params || {});
        } catch (error) {
          fail(`  (CDP listener for ${message.method} threw: ${error.message})`);
        }
      }
    });

    const rejectAll = (reason) => {
      this.closed = true;
      for (const pending of this.pending.values()) {
        pending.reject(new Error(reason));
      }
      this.pending.clear();
    };
    socket.addEventListener("close", () => rejectAll("DevTools WebSocket closed"));
    socket.addEventListener("error", () => rejectAll("DevTools WebSocket error"));
  }

  on(method, listener) {
    if (!this.listeners.has(method)) this.listeners.set(method, new Set());
    const listeners = this.listeners.get(method);
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  send(method, params = {}) {
    if (this.closed) return Promise.reject(new Error(`Cannot send ${method}: CDP socket is closed`));
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, method });
      try {
        this.socket.send(JSON.stringify({ id, method, params }));
      } catch (error) {
        this.pending.delete(id);
        reject(error);
      }
    });
  }

  close() {
    try {
      this.socket.close();
    } catch {
      /* already closed */
    }
  }
}

async function connectCdp(wsUrl, timeoutMs = 10000) {
  const socket = new WebSocket(normalizeWsUrl(wsUrl));
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      socket.close();
      reject(new Error(`Timed out connecting to DevTools WebSocket: ${wsUrl}`));
    }, timeoutMs);
    socket.addEventListener(
      "open",
      () => {
        clearTimeout(timer);
        resolve();
      },
      { once: true },
    );
    socket.addEventListener(
      "error",
      () => {
        clearTimeout(timer);
        reject(new Error(`Failed to connect to DevTools WebSocket: ${wsUrl}`));
      },
      { once: true },
    );
  });
  return new CDP(socket);
}

function withTimeout(promise, timeoutMs, message) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(message)), timeoutMs);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

async function launchChrome(chromePath, userDataDir) {
  const args = [
    "--headless=new",
    "--no-sandbox",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--remote-debugging-port=" + PORT,
    "--remote-debugging-address=" + HOST,
    "--user-data-dir=" + userDataDir,
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-sync",
    "--mute-audio",
    "--hide-crash-restore-bubble",
    "--window-size=" + VIEWPORTS[0].width + "," + VIEWPORTS[0].height,
    "about:blank",
  ];

  const child = spawn(chromePath, args, { stdio: ["ignore", "ignore", "pipe"] });
  const state = { child, stderr: "", exit: null, spawnError: null };

  state.done = new Promise((resolve) => {
    child.once("error", (error) => {
      state.spawnError = error;
      resolve(false);
    });
    child.once("exit", (code, signal) => {
      state.exit = { code, signal };
      resolve(true);
    });
  });

  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (chunk) => {
    state.stderr = (state.stderr + chunk).slice(-24000);
  });

  return state;
}

function chromeDiagnostics(state) {
  const lines = [];
  if (state.exit) {
    lines.push(`Chrome exited (code=${state.exit.code}, signal=${state.exit.signal}).`);
  }
  if (state.spawnError) lines.push(`Chrome spawn error: ${state.spawnError.message}`);
  const tail = state.stderr.trim();
  if (tail) {
    lines.push("Chrome stderr (tail):", tail.split("\n").slice(-20).join("\n"));
  }
  return lines.join("\n");
}

async function waitForDevTools(state, deadline) {
  while (Date.now() < deadline) {
    if (state.spawnError) {
      throw new Error(`Failed to launch Chrome: ${state.spawnError.message}\n${chromeDiagnostics(state)}`);
    }
    if (state.exit) {
      throw new Error(`Chrome exited before DevTools was ready.\n${chromeDiagnostics(state)}`);
    }
    try {
      return await fetchJson(`http://${HOST}:${PORT}/json/version`);
    } catch {
      /* Chrome is still starting. */
    }
    await sleep(120);
  }
  throw new Error(
    `Timed out waiting for Chrome's DevTools endpoint at http://${HOST}:${PORT}/json/version.\n` +
      chromeDiagnostics(state),
  );
}

async function waitForPageTarget(state, deadline) {
  while (Date.now() < deadline) {
    if (state.exit) {
      throw new Error(`Chrome exited before a page target existed.\n${chromeDiagnostics(state)}`);
    }
    try {
      const targets = await fetchJson(`http://${HOST}:${PORT}/json/list`);
      const page = targets.find(
        (target) => target.type === "page" && target.webSocketDebuggerUrl,
      );
      if (page) return page;
    } catch {
      /* Retry until the deadline. */
    }
    await sleep(120);
  }
  throw new Error(
    `Timed out waiting for a Chrome page target.\n${chromeDiagnostics(state)}`,
  );
}

function formatException(details) {
  const description =
    details?.exception?.description ||
    details?.exception?.value ||
    details?.text ||
    "Unknown exception";
  const location = details?.url
    ? `${details.url}:${(details.lineNumber ?? 0) + 1}:${(details.columnNumber ?? 0) + 1}`
    : "";
  return { text: String(description), detail: location };
}

function formatConsoleArgs(params) {
  return (params.args || [])
    .map((arg) => {
      if (arg.type === "string") return arg.value;
      if (Object.prototype.hasOwnProperty.call(arg, "value")) return String(arg.value);
      return arg.description || `<${arg.type}>`;
    })
    .join(" ");
}

function createErrorCollector() {
  return {
    errors: [],
    reset() {
      this.errors = [];
    },
    push(entry) {
      this.errors.push(entry);
    },
  };
}

function dedupeErrors(errors) {
  const seen = new Set();
  return errors.filter((error) => {
    const key = `${error.source}|${error.text}|${error.detail}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function short(value, max = 300) {
  const text = String(value).replace(/\s+/g, " ").trim();
  return text.length > max ? text.slice(0, max - 1) + "…" : text;
}

function printErrors(errors) {
  const unique = dedupeErrors(errors);
  for (const error of unique.slice(0, 8)) {
    const where = error.detail ? ` (${short(error.detail, 160)})` : "";
    fail(`    [${error.source}] ${short(error.text, 500)}${where}`);
  }
  if (unique.length > 8) fail(`    … ${unique.length - 8} more console error(s)`);
}

async function evaluate(cdp, expression, timeoutMs = 15000) {
  const response = await withTimeout(
    cdp.send("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
      userGesture: false,
    }),
    timeoutMs,
    `Runtime.evaluate timed out after ${timeoutMs}ms`,
  );
  if (response.exceptionDetails) {
    const exception = formatException(response.exceptionDetails);
    throw new Error(`Page evaluation failed: ${exception.text}${exception.detail ? ` (${exception.detail})` : ""}`);
  }
  return response.result?.value;
}

/** Evaluate after the page has finished navigating, retrying destroyed contexts. */
async function evaluateWhenReady(cdp, expression, timeoutMs = PAGE_LOAD_TIMEOUT_MS) {
  const deadline = Date.now() + timeoutMs;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      return await evaluate(cdp, expression, Math.max(1000, deadline - Date.now()));
    } catch (error) {
      lastError = error;
      await sleep(100);
    }
  }
  throw new Error(`Page did not become evaluable within ${timeoutMs}ms: ${lastError?.message || "unknown error"}`);
}

function waitForEvent(cdp, method, timeoutMs) {
  let cancel = () => {};
  const promise = new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      off();
      reject(new Error(`Timed out waiting for ${method} after ${timeoutMs}ms`));
    }, timeoutMs);
    const off = cdp.on(method, (params) => {
      clearTimeout(timer);
      off();
      resolve(params);
    });
    cancel = () => {
      clearTimeout(timer);
      off();
    };
  });
  return { promise, cancel };
}

async function enableDomains(cdp) {
  await Promise.all([
    cdp.send("Page.enable"),
    cdp.send("Runtime.enable"),
    cdp.send("Log.enable"),
  ]);
}

function installErrorListeners(cdp, collector) {
  cdp.on("Runtime.exceptionThrown", (params) => {
    const exception = formatException(params.exceptionDetails);
    collector.push({ source: "exception", text: exception.text, detail: exception.detail });
  });
  cdp.on("Log.entryAdded", (params) => {
    const entry = params.entry || {};
    if (entry.level !== "error") return;
    collector.push({
      source: "log:" + (entry.source || "unknown"),
      text: entry.text || "(empty log message)",
      detail: entry.url || "",
    });
  });
  cdp.on("Runtime.consoleAPICalled", (params) => {
    if (params.type !== "error") return;
    collector.push({
      source: "console.error",
      text: formatConsoleArgs(params),
      detail: (params.stackTrace?.callFrames || [])
        .slice(0, 1)
        .map((frame) => `${frame.url || ""}:${(frame.lineNumber ?? 0) + 1}`)
        .join(""),
    });
  });
}

async function loadPage(cdp, collector, viewport) {
  collector.reset();
  await cdp.send("Emulation.setDeviceMetricsOverride", {
    width: viewport.width,
    height: viewport.height,
    deviceScaleFactor: 1,
    mobile: false,
    screenWidth: viewport.width,
    screenHeight: viewport.height,
  });
  const loaded = waitForEvent(cdp, "Page.loadEventFired", PAGE_LOAD_TIMEOUT_MS);
  try {
    const navigation = await cdp.send("Page.navigate", { url: PAGE_URL });
    if (navigation.errorText) {
      throw new Error(`Page.navigate failed: ${navigation.errorText}`);
    }
    await loaded.promise;
  } catch (error) {
    loaded.cancel();
    throw error;
  }

  return evaluateWhenReady(
    cdp,
    `(async () => {
      const deadline = performance.now() + ${PAGE_LOAD_TIMEOUT_MS - 2000};
      while (document.readyState !== "complete" || !document.body) {
        if (performance.now() > deadline) {
          return { ok: false, readyState: document.readyState };
        }
        await new Promise((resolve) => setTimeout(resolve, 25));
      }
      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      await new Promise((resolve) => setTimeout(resolve, ${SETTLE_MS}));
      const active = document.querySelectorAll("section.view.active");
      return {
        ok: true,
        view: document.body.dataset.view || null,
        hash: location.hash,
        activeCount: active.length,
        activeIds: Array.from(active).map((section) => section.id),
        scrollWidth: document.documentElement.scrollWidth,
        innerWidth: window.innerWidth,
        readyState: document.readyState,
      };
    })()`,
  );
}

async function runRoute(cdp, route, viewport) {
  const expectedView = route.slice(1).split("/")[0];
  const expression = `(async () => {
    const targetHash = ${JSON.stringify(route)};
    const expectedView = ${JSON.stringify(expectedView)};
    if (location.hash !== targetHash) {
      location.hash = targetHash;
    }
    const deadline = performance.now() + ${ROUTE_TIMEOUT_MS - 1000};
    while ((document.body?.dataset.view || null) !== expectedView) {
      if (performance.now() > deadline) {
        const active = document.querySelectorAll("section.view.active");
        return {
          ok: false,
          reason: "view-timeout",
          expectedView,
          view: document.body?.dataset.view || null,
          hash: location.hash,
          activeCount: active.length,
          activeIds: Array.from(active).map((section) => section.id),
          scrollWidth: document.documentElement.scrollWidth,
          innerWidth: window.innerWidth,
        };
      }
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    await new Promise((resolve) => setTimeout(resolve, ${SETTLE_MS}));
    const active = Array.from(document.querySelectorAll("section.view.active"));
    return {
      ok: true,
      expectedView,
      view: document.body.dataset.view,
      hash: location.hash,
      activeCount: active.length,
      activeIds: active.map((section) => section.id),
      scrollWidth: document.documentElement.scrollWidth,
      innerWidth: window.innerWidth,
      readyState: document.readyState,
    };
  })()`;

  return evaluate(cdp, expression, ROUTE_TIMEOUT_MS + 4000);
}

async function runViewport(cdp, viewport, collector) {
  const results = [];
  const failures = [];

  try {
    const initial = await loadPage(cdp, collector, viewport);
    const initialErrors = collector.errors;
    if (!initial?.ok) {
      throw new Error(
        `initial page load did not finish (readyState=${initial?.readyState || "unknown"})`,
      );
    }
    if (initialErrors.length) {
      failures.push({
        route: "(initial load)",
        viewport,
        errors: initialErrors,
      });
      fail(`FAIL · ${viewport.name} (initial load): ${initialErrors.length} console error(s)`);
      printErrors(initialErrors);
    }
  } catch (error) {
    fail(`FAIL · ${viewport.name} (initial load): ${error.message}`);
    collector.push({ source: "harness", text: error.message, detail: "" });
    return { results, failures: [{ route: "(initial load)", viewport, errors: collector.errors, error }] };
  }

  for (const route of ROUTES) {
    collector.reset();
    let state = null;
    try {
      state = await runRoute(cdp, route, viewport);
    } catch (error) {
      failures.push({ route, viewport, errors: collector.errors, error });
      fail(`FAIL · ${viewport.name} ${route}: ${error.message}`);
      if (collector.errors.length) printErrors(collector.errors);
      if (cdp.closed) break;
      continue;
    }

    const errors = collector.errors;
    const problems = [];
    if (!state?.ok) {
      problems.push(
        `body[data-view] never became "${expectedViewFor(route)}" (got ${JSON.stringify(state?.view ?? null)}, hash=${state?.hash ?? "?"})`,
      );
    } else {
      if (state.activeCount !== 1) {
        problems.push(
          `expected exactly 1 section.view.active, got ${state.activeCount}` +
            (state.activeIds?.length ? ` (${state.activeIds.join(", ")})` : ""),
        );
      }
      if (state.view !== expectedViewFor(route)) {
        problems.push(
          `body[data-view]="${state.view}" does not match route "${expectedViewFor(route)}"`,
        );
      }
      if (state.readyState !== "complete") {
        problems.push(`document.readyState="${state.readyState}" (expected "complete")`);
      }
      const overflow = state.scrollWidth - state.innerWidth;
      if (overflow > 1) {
        problems.push(
          `horizontal overflow: documentElement.scrollWidth=${state.scrollWidth} > innerWidth=${state.innerWidth} + 1`,
        );
      }
    }
    if (errors.length) {
      problems.push(`${errors.length} console error(s)`);
    }

    results.push({ route, state, errors, problems });
    if (problems.length) {
      failures.push({ route, viewport, errors, state, problems });
      fail(`FAIL · ${viewport.name} ${route}: ${problems.join("; ")}`);
      if (errors.length) printErrors(errors);
    } else {
      const overflowPx = Math.max(0, state.scrollWidth - state.innerWidth);
      log(
        `[${viewport.name}] ${route.padEnd(14)} view=${String(state.view).padEnd(11)} ` +
          `active=${state.activeCount} errors=0 overflow=${overflowPx}px`,
      );
    }
  }

  return { results, failures };
}

function expectedViewFor(route) {
  return route.slice(1).split("/")[0];
}

function validateRoutes() {
  const knownViews = new Set([
    "think", "kv", "prefix", "paged", "transformer", "anatomy", "calc",
    "train", "net", "nccl", "cuda", "ecn", "gpu", "npu", "opt", "feedback",
  ]);
  for (const route of ROUTES) {
    const view = expectedViewFor(route);
    if (!knownViews.has(view)) {
      throw new Error(`ROUTES contains unknown view "${view}" in ${route}`);
    }
  }
}

async function closeBrowser(instance) {
  const { state, browserCdp, pageCdp } = instance;
  if (!state?.child || state.exit) return;

  for (const client of [browserCdp, pageCdp]) {
    if (!client || client.closed) continue;
    try {
      await withTimeout(client.send("Browser.close"), 2000, "Browser.close timed out");
      break;
    } catch {
      /* The browser may close the socket before replying; verify below. */
    }
  }

  const exited = await withTimeout(state.done, 5000, "Chrome did not exit within 5s").catch(
    () => false,
  );
  if (exited || state.exit) return;

  for (const signal of ["SIGTERM", "SIGKILL"]) {
    try {
      state.child.kill(signal);
    } catch {
      /* ignore */
    }
    if (await withTimeout(state.done, 1500, "Chrome did not exit").catch(() => false)) return;
  }
}

async function removeTempDir(dir) {
  if (!dir) return;
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      await fs.promises.rm(dir, { recursive: true, force: true, maxRetries: 2 });
      return;
    } catch {
      await sleep(150);
    }
  }
  fail(`warning: could not remove temporary Chrome profile: ${dir}`);
}

async function main() {
  if (typeof fetch !== "function" || typeof WebSocket !== "function") {
    throw new Error("This smoke test requires Node 22+ (global fetch and WebSocket).");
  }
  validateRoutes();
  if (!fs.existsSync(path.join(REPO_ROOT, "web", "index.html"))) {
    throw new Error(`web/index.html not found under ${REPO_ROOT}`);
  }

  const chromePath = findChrome();
  await assertPortFree(PORT);

  const userDataDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "whykvc-smoke-"));
  const instance = { state: null, browserCdp: null, pageCdp: null, userDataDir };

  log(`whykvc smoke · ${PAGE_URL}`);
  log(`chrome: ${chromePath}`);
  log(`viewports: ${VIEWPORTS.map((v) => v.name).join(", ")} · ${ROUTES.length} routes`);

  try {
    instance.state = await launchChrome(chromePath, userDataDir);
    const deadline = Date.now() + 30000;
    const version = await waitForDevTools(instance.state, deadline);
    if (!version.webSocketDebuggerUrl) {
      throw new Error("Chrome /json/version did not return a webSocketDebuggerUrl.");
    }
    instance.browserCdp = await connectCdp(version.webSocketDebuggerUrl);

    const target = await waitForPageTarget(instance.state, deadline);
    if (!target.webSocketDebuggerUrl) {
      throw new Error("Chrome /json/list did not return a page webSocketDebuggerUrl.");
    }
    instance.pageCdp = await connectCdp(target.webSocketDebuggerUrl);
    const cdp = instance.pageCdp;

    await enableDomains(cdp);
    const collector = createErrorCollector();
    installErrorListeners(cdp, collector);

    const allFailures = [];
    for (const viewport of VIEWPORTS) {
      const { results, failures } = await runViewport(cdp, viewport, collector);
      allFailures.push(...failures);
      if (failures.length) {
        fail(`\n${viewport.name}: ${failures.length} failing route(s) of ${ROUTES.length}`);
      } else {
        log(`[${viewport.name}] PASS ${results.length}/${ROUTES.length} routes · 0 console errors · no horizontal overflow`);
      }
      if (cdp.closed) {
        allFailures.push({
          route: "(all remaining viewports)",
          viewport,
          errors: [],
          error: new Error("DevTools disconnected"),
        });
        break;
      }
    }

    if (allFailures.length) {
      fail("");
      fail(`FAIL · ${allFailures.length} failure(s)`);
      for (const failure of allFailures) {
        const label = `${failure.viewport?.name || "?"} ${failure.route}`;
        if (failure.problems) {
          fail(`  - ${label}: ${failure.problems.join("; ")}`);
        } else {
          fail(`  - ${label}: ${failure.error?.message || "failed"}`);
        }
      }
      process.exitCode = 1;
      return;
    }

    log("");
    log(
      `PASS · ${VIEWPORTS.length} viewports × ${ROUTES.length} routes · 0 console errors · no horizontal overflow`,
    );
  } finally {
    await closeBrowser(instance);
    instance.browserCdp?.close();
    instance.pageCdp?.close();
    await removeTempDir(userDataDir);
  }
}

main().catch((error) => {
  fail("");
  fail(`FAIL · ${error?.message || error}`);
  if (process.env.SMOKE_DEBUG) fail(error?.stack || String(error));
  process.exitCode = 1;
});
