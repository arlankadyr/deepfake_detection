(() => {
  const modelLabels = { resnet50: 'ResNet-50', deit_small: 'DeiT-S' };
  const modelColors = { resnet50: '--accent', deit_small: '--blue' };
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const status = document.getElementById('results-status');
  const emptyState = document.getElementById('no-results');
  const dashboard = document.getElementById('results-dashboard');
  const modelFilter = document.getElementById('filter-model');
  const testFilter = document.getElementById('filter-test-set');
  const sizeFilter = document.getElementById('filter-size');
  let runs = [];

  function modelName(value) { return modelLabels[value] || value; }
  function asNumber(value) { const result = Number(value); return Number.isFinite(result) ? result : null; }
  function formatMetric(value) { return value == null ? '—' : value.toFixed(3); }
  function formatSize(value) { return Number(value).toLocaleString('en-US'); }
  function meanAndDeviation(values) {
    if (!values.length) return { mean: null, deviation: null, count: 0 };
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.length > 1
      ? values.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / (values.length - 1)
      : 0;
    return { mean, deviation: Math.sqrt(variance), count: values.length };
  }

  function makeOption(value, label) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    return option;
  }

  function initializeFilters() {
    const models = [...new Set(runs.map((run) => run.model))].sort();
    const testSets = [...new Set(runs.map((run) => run.test_set))].sort();
    const sizes = [...new Set(runs.map((run) => run.train_size))].sort((a, b) => b - a);
    modelFilter.append(...models.map((value) => makeOption(value, modelName(value))));
    testFilter.append(...testSets.map((value) => makeOption(value, value)));
    sizeFilter.append(...sizes.map((value) => makeOption(String(value), formatSize(value))));
  }

  function filteredRuns(ignoreSize = false) {
    return runs.filter((run) =>
      (modelFilter.value === 'all' || run.model === modelFilter.value) &&
      (testFilter.value === 'all' || run.test_set === testFilter.value) &&
      (ignoreSize || sizeFilter.value === 'all' || run.train_size === Number(sizeFilter.value))
    );
  }

  function appendCell(row, value, className) {
    const cell = document.createElement('td');
    cell.textContent = value;
    if (className) cell.className = className;
    row.append(cell);
  }

  function renderRunTable(items) {
    const body = document.getElementById('runs-body');
    body.replaceChildren();
    items.forEach((run) => {
      const row = document.createElement('tr');
      appendCell(row, modelName(run.model));
      appendCell(row, formatSize(run.train_size));
      appendCell(row, String(run.seed));
      appendCell(row, run.test_set);
      appendCell(row, formatMetric(run.auc), 'metric-number');
      appendCell(row, formatMetric(run.accuracy), 'metric-number');
      appendCell(row, formatMetric(run.f1), 'metric-number');
      body.append(row);
    });
    document.getElementById('run-count').textContent = `${items.length} run${items.length === 1 ? '' : 's'}`;
  }

  function groupBy(items, keyFunction, valueFunction) {
    const groups = new Map();
    items.forEach((item) => {
      const key = keyFunction(item);
      if (!groups.has(key)) groups.set(key, []);
      const value = valueFunction(item);
      if (value != null) groups.get(key).push(value);
    });
    return groups;
  }

  function renderSummaryTable(items) {
    const body = document.getElementById('summary-body');
    body.replaceChildren();
    const groups = groupBy(
      items,
      (run) => `${run.model}\u001f${run.train_size}\u001f${run.test_set}`,
      (run) => run.auc,
    );
    [...groups.entries()].sort((left, right) => left[0].localeCompare(right[0], undefined, { numeric: true })).forEach(([key, values]) => {
      const [model, size, testSet] = key.split('\u001f');
      const stats = meanAndDeviation(values);
      const row = document.createElement('tr');
      appendCell(row, modelName(model));
      appendCell(row, formatSize(size));
      appendCell(row, testSet);
      appendCell(row, `${formatMetric(stats.mean)} ± ${formatMetric(stats.deviation)}`, 'metric-number');
      appendCell(row, String(stats.count));
      body.append(row);
    });
  }

  function svgElement(tag, attributes = {}, content = null) {
    const element = document.createElementNS(SVG_NS, tag);
    Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, String(value)));
    if (content != null) element.textContent = content;
    return element;
  }

  function cssColor(variable) {
    return getComputedStyle(document.documentElement).getPropertyValue(variable).trim();
  }

  function showChartEmpty(container, message = 'No matching experiment data.') {
    container.replaceChildren();
    const empty = document.createElement('div');
    empty.className = 'chart-empty';
    empty.textContent = message;
    container.append(empty);
  }

  function updateLegend(containerId, series) {
    const container = document.getElementById(containerId);
    container.replaceChildren();
    series.forEach((name) => {
      const item = document.createElement('span');
      item.className = 'legend-item';
      const swatch = document.createElement('span');
      swatch.className = 'legend-swatch';
      swatch.style.backgroundColor = cssColor(modelColors[name] || '--ink-faint');
      const label = document.createElement('span');
      label.textContent = modelName(name);
      item.append(swatch, label);
      container.append(item);
    });
  }

  function labelLines(label) {
    const words = String(label).split(' ');
    if (words.length === 1 && label.length > 15) return [label.slice(0, 13), `${label.slice(13, 25)}${label.length > 25 ? '…' : ''}`];
    if (words.length > 2) return [words.slice(0, 2).join(' '), words.slice(2).join(' ')];
    return [label];
  }

  function renderBarChart(container, categories, series, valuesByCategory, yMin, yMax, yLabel) {
    if (!categories.length || !series.length) {
      showChartEmpty(container);
      return;
    }
    const width = Math.max(700, categories.length * 112);
    const height = 250;
    const margin = { top: 15, right: 20, bottom: 54, left: 49 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    const y = (value) => margin.top + ((yMax - value) / (yMax - yMin)) * chartHeight;
    const zeroY = y(Math.max(yMin, Math.min(yMax, 0)));
    const svg = svgElement('svg', { viewBox: `0 0 ${width} ${height}`, role: 'presentation', focusable: 'false' });

    for (let step = 0; step <= 4; step += 1) {
      const value = yMin + ((yMax - yMin) * step) / 4;
      const yPos = y(value);
      svg.append(svgElement('line', { x1: margin.left, x2: width - margin.right, y1: yPos, y2: yPos, stroke: cssColor('--line'), 'stroke-width': 1 }));
      svg.append(svgElement('text', { x: margin.left - 8, y: yPos + 3, 'text-anchor': 'end', fill: cssColor('--ink-faint'), 'font-size': 12, 'font-family': 'DM Mono, monospace' }, value.toFixed(2)));
    }
    svg.append(svgElement('line', { x1: margin.left, x2: margin.left, y1: margin.top, y2: height - margin.bottom, stroke: cssColor('--line-strong'), 'stroke-width': 1 }));
    svg.append(svgElement('line', { x1: margin.left, x2: width - margin.right, y1: zeroY, y2: zeroY, stroke: cssColor('--line-strong'), 'stroke-width': 1 }));
    const groupWidth = chartWidth / categories.length;
    const barWidth = Math.max(4, Math.min(23, (groupWidth * 0.66) / series.length));

    categories.forEach((category, categoryIndex) => {
      const categoryValues = valuesByCategory.get(category.key) || new Map();
      const center = margin.left + groupWidth * (categoryIndex + 0.5);
      const totalWidth = barWidth * series.length;
      series.forEach((model, seriesIndex) => {
        const stats = categoryValues.get(model);
        if (!stats || stats.mean == null) return;
        const x = center - totalWidth / 2 + seriesIndex * barWidth + barWidth * 0.1;
        const valueY = y(Math.max(yMin, Math.min(yMax, stats.mean)));
        const rectY = Math.min(zeroY, valueY);
        const rectHeight = Math.max(1, Math.abs(zeroY - valueY));
        const color = cssColor(modelColors[model] || '--ink-faint');
        svg.append(svgElement('rect', { x, y: rectY, width: barWidth * 0.8, height: rectHeight, rx: 2, fill: color, opacity: .88 }));
        if (stats.deviation > 0) {
          const capX = x + barWidth * .4;
          const lowY = y(Math.max(yMin, Math.min(yMax, stats.mean - stats.deviation)));
          const highY = y(Math.max(yMin, Math.min(yMax, stats.mean + stats.deviation)));
          svg.append(svgElement('line', { x1: capX, x2: capX, y1: lowY, y2: highY, stroke: color, 'stroke-width': 1.3 }));
          svg.append(svgElement('line', { x1: capX - 3, x2: capX + 3, y1: lowY, y2: lowY, stroke: color, 'stroke-width': 1.3 }));
          svg.append(svgElement('line', { x1: capX - 3, x2: capX + 3, y1: highY, y2: highY, stroke: color, 'stroke-width': 1.3 }));
        }
      });
      const label = svgElement('text', { x: center, y: height - margin.bottom + 15, 'text-anchor': 'middle', fill: cssColor('--ink-faint'), 'font-size': 12, 'font-family': 'DM Sans, sans-serif' });
      labelLines(category.label).forEach((line, lineIndex) => label.append(svgElement('tspan', { x: center, dy: lineIndex === 0 ? 0 : 11 }, line)));
      svg.append(label);
    });
    const axisLabel = svgElement('text', { x: 12, y: margin.top + chartHeight / 2, transform: `rotate(-90 12 ${margin.top + chartHeight / 2})`, 'text-anchor': 'middle', fill: cssColor('--ink-faint'), 'font-size': 12, 'font-family': 'DM Sans, sans-serif' }, yLabel);
    svg.append(axisLabel);
    container.replaceChildren(svg);
  }

  function statsBy(items, keyFunction, valueFunction) {
    const rawGroups = groupBy(items, keyFunction, valueFunction);
    const result = new Map();
    rawGroups.forEach((values, key) => result.set(key, meanAndDeviation(values)));
    return result;
  }

  function renderAucChart(items) {
    const container = document.getElementById('auc-chart');
    const usable = items.filter((run) => run.auc != null);
    const groups = statsBy(usable, (run) => `${run.test_set}\u001f${run.train_size}\u001f${run.model}`, (run) => run.auc);
    const keys = [...new Set(usable.map((run) => `${run.test_set}\u001f${run.train_size}`))].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
    const categories = keys.map((key) => {
      const [testSet, size] = key.split('\u001f');
      return { key, label: `${testSet} · n=${formatSize(size)}` };
    });
    const series = [...new Set(usable.map((run) => run.model))].sort();
    const values = new Map(categories.map((category) => {
      const grouped = new Map();
      series.forEach((model) => {
        const stats = groups.get(`${category.key}\u001f${model}`);
        if (stats) grouped.set(model, stats);
      });
      return [category.key, grouped];
    }));
    updateLegend('auc-legend', series);
    renderBarChart(container, categories, series, values, 0, 1, 'AUC');
  }

  function matchedDrops() {
    const baselineByRun = new Map();
    runs.filter((run) => run.test_set.toLowerCase() === 'biggan' && run.auc != null).forEach((run) => {
      baselineByRun.set(`${run.model}\u001f${run.train_size}\u001f${run.seed}`, run.auc);
    });
    const drops = [];
    runs.filter((run) => run.test_set.toLowerCase() !== 'biggan' && run.auc != null).forEach((run) => {
      const baseline = baselineByRun.get(`${run.model}\u001f${run.train_size}\u001f${run.seed}`);
      if (baseline == null) return;
      drops.push({ ...run, auc_drop: baseline - run.auc });
    });
    return drops;
  }

  function renderDropChart(items) {
    const container = document.getElementById('drop-chart');
    const allowedKeys = new Set(items.map((run) => `${run.model}\u001f${run.train_size}\u001f${run.test_set}`));
    const usable = matchedDrops().filter((run) => allowedKeys.has(`${run.model}\u001f${run.train_size}\u001f${run.test_set}`));
    const groups = statsBy(usable, (run) => `${run.test_set}\u001f${run.train_size}\u001f${run.model}`, (run) => run.auc_drop);
    const keys = [...new Set(usable.map((run) => `${run.test_set}\u001f${run.train_size}`))].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
    const categories = keys.map((key) => {
      const [testSet, size] = key.split('\u001f');
      return { key, label: `${testSet} · n=${formatSize(size)}` };
    });
    const series = [...new Set(usable.map((run) => run.model))].sort();
    const values = new Map(categories.map((category) => {
      const grouped = new Map();
      series.forEach((model) => {
        const stats = groups.get(`${category.key}\u001f${model}`);
        if (stats) grouped.set(model, stats);
      });
      return [category.key, grouped];
    }));
    updateLegend('drop-legend', series);
    const allValues = [...groups.values()].flatMap((stats) => [stats.mean - stats.deviation, stats.mean + stats.deviation]);
    const low = Math.min(0, ...allValues);
    const high = Math.max(0, ...allValues);
    const pad = high === low ? .05 : (high - low) * .12;
    renderBarChart(container, categories, series, values, low - pad, high + pad, 'AUC drop');
  }

  function renderLearningCurve(items) {
    const container = document.getElementById('learning-chart');
    const drops = matchedDrops().filter((run) =>
      (modelFilter.value === 'all' || run.model === modelFilter.value) &&
      (testFilter.value === 'all' || run.test_set === testFilter.value)
    );
    const models = [...new Set(drops.map((run) => run.model))].sort();
    const sizes = [...new Set(drops.map((run) => run.train_size))].sort((a, b) => a - b);
    updateLegend('learning-legend', models);
    if (!sizes.length || !models.length) {
      showChartEmpty(container, 'Learning curve appears when matched BigGAN and diffusion results are available.');
      return;
    }

    const width = Math.max(700, sizes.length * 100);
    const height = 285;
    const margin = { top: 18, right: 24, bottom: 51, left: 53 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    const grouped = new Map();
    models.forEach((model) => {
      const bySize = new Map();
      sizes.forEach((size) => {
        const stats = meanAndDeviation(drops.filter((run) => run.model === model && run.train_size === size).map((run) => run.auc_drop));
        if (stats.count) bySize.set(size, stats);
      });
      grouped.set(model, bySize);
    });
    const allStats = [...grouped.values()].flatMap((entries) => [...entries.values()]);
    let yMin = Math.min(0, ...allStats.map((stats) => stats.mean - stats.deviation));
    let yMax = Math.max(0, ...allStats.map((stats) => stats.mean + stats.deviation));
    const padding = yMax === yMin ? .05 : (yMax - yMin) * .12;
    yMin -= padding;
    yMax += padding;
    const y = (value) => margin.top + ((yMax - value) / (yMax - yMin)) * chartHeight;
    const xMin = Math.min(...sizes);
    const xMax = Math.max(...sizes);
    const x = (size) => xMin === xMax ? margin.left + chartWidth / 2 : margin.left + ((size - xMin) / (xMax - xMin)) * chartWidth;
    const svg = svgElement('svg', { viewBox: `0 0 ${width} ${height}`, role: 'presentation', focusable: 'false' });

    for (let step = 0; step <= 4; step += 1) {
      const value = yMin + ((yMax - yMin) * step) / 4;
      const yPos = y(value);
      svg.append(svgElement('line', { x1: margin.left, x2: width - margin.right, y1: yPos, y2: yPos, stroke: cssColor('--line'), 'stroke-width': 1 }));
      svg.append(svgElement('text', { x: margin.left - 8, y: yPos + 3, 'text-anchor': 'end', fill: cssColor('--ink-faint'), 'font-size': 12, 'font-family': 'DM Mono, monospace' }, value.toFixed(2)));
    }
    svg.append(svgElement('line', { x1: margin.left, x2: width - margin.right, y1: y(0), y2: y(0), stroke: cssColor('--line-strong'), 'stroke-width': 1 }));
    sizes.forEach((size) => {
      const xPos = x(size);
      svg.append(svgElement('line', { x1: xPos, x2: xPos, y1: height - margin.bottom, y2: height - margin.bottom + 4, stroke: cssColor('--line-strong'), 'stroke-width': 1 }));
      svg.append(svgElement('text', { x: xPos, y: height - margin.bottom + 17, 'text-anchor': 'middle', fill: cssColor('--ink-faint'), 'font-size': 12, 'font-family': 'DM Mono, monospace' }, formatSize(size)));
    });
    svg.append(svgElement('line', { x1: margin.left, x2: width - margin.right, y1: height - margin.bottom, y2: height - margin.bottom, stroke: cssColor('--line-strong'), 'stroke-width': 1 }));
    svg.append(svgElement('text', { x: margin.left + chartWidth / 2, y: height - 8, 'text-anchor': 'middle', fill: cssColor('--ink-faint'), 'font-size': 12, 'font-family': 'DM Sans, sans-serif' }, 'Training-set size'));

    models.forEach((model) => {
      const points = [...grouped.get(model).entries()].sort((left, right) => left[0] - right[0]);
      const color = cssColor(modelColors[model] || '--ink-faint');
      const path = points.map(([size, stats], index) => `${index === 0 ? 'M' : 'L'} ${x(size)} ${y(stats.mean)}`).join(' ');
      svg.append(svgElement('path', { d: path, fill: 'none', stroke: color, 'stroke-width': 2.4, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }));
      points.forEach(([size, stats]) => {
        const xPos = x(size);
        const yPos = y(stats.mean);
        if (stats.deviation > 0) {
          const lowY = y(stats.mean - stats.deviation);
          const highY = y(stats.mean + stats.deviation);
          svg.append(svgElement('line', { x1: xPos, x2: xPos, y1: lowY, y2: highY, stroke: color, 'stroke-width': 1.2 }));
          svg.append(svgElement('line', { x1: xPos - 3, x2: xPos + 3, y1: lowY, y2: lowY, stroke: color, 'stroke-width': 1.2 }));
          svg.append(svgElement('line', { x1: xPos - 3, x2: xPos + 3, y1: highY, y2: highY, stroke: color, 'stroke-width': 1.2 }));
        }
        svg.append(svgElement('circle', { cx: xPos, cy: yPos, r: 4, fill: color, stroke: cssColor('--surface'), 'stroke-width': 2 }));
      });
    });
    container.replaceChildren(svg);
  }

  function renderDashboard() {
    const selected = filteredRuns();
    renderRunTable(selected);
    renderSummaryTable(selected);
    renderAucChart(selected);
    renderDropChart(selected);
    renderLearningCurve(runs);
  }

  async function loadResults() {
    try {
      const response = await fetch('/api/results');
      if (response.status === 404) {
        status.textContent = 'No experiment results yet.';
        emptyState.hidden = false;
        return;
      }
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Experiment results could not be loaded.');
      if (!data || !Array.isArray(data.runs)) throw new Error('The metrics file does not contain a valid runs list.');
      runs = data.runs.map((run) => ({
        model: String(run.model ?? 'unknown'),
        train_size: asNumber(run.train_size),
        seed: asNumber(run.seed),
        test_set: String(run.test_set ?? 'unknown'),
        auc: asNumber(run.auc),
        accuracy: asNumber(run.accuracy),
        f1: asNumber(run.f1),
      })).filter((run) => run.train_size != null && run.seed != null);
      initializeFilters();
      dashboard.hidden = false;
      status.textContent = `${runs.length} reported run${runs.length === 1 ? '' : 's'}`;
      renderDashboard();
    } catch (error) {
      status.textContent = error.message || 'Experiment results could not be loaded.';
      emptyState.hidden = false;
      emptyState.querySelector('h3').textContent = 'Results could not be loaded';
      emptyState.querySelector('p').textContent = error.message || 'Check the experiment metrics file and reload this page.';
    }
  }

  [modelFilter, testFilter, sizeFilter].forEach((filter) => filter.addEventListener('change', renderDashboard));
  loadResults();
})();
