(function () {
  const root = document.documentElement;
  const savedTheme = localStorage.getItem("diabetes-ai-theme") || "light";
  root.setAttribute("data-theme", savedTheme);

  function updateThemeButtons(theme) {
    document.querySelectorAll(".theme-toggle").forEach((button) => {
      const icon = button.querySelector("i");
      if (icon) icon.className = theme === "dark" ? "bi bi-sun me-2" : "bi bi-moon-stars me-2";
      const textNodes = Array.from(button.childNodes).filter((node) => node.nodeType === Node.TEXT_NODE);
      if (textNodes.length) textNodes[textNodes.length - 1].textContent = theme === "dark" ? " Light Mode" : " Dark Mode";
    });
  }

  function initTooltips() {
    if (!window.bootstrap) return;
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach((el) => new bootstrap.Tooltip(el));
  }

  function initToasts() {
    if (!window.bootstrap) return;
    document.querySelectorAll(".app-toast").forEach((toastEl) => new bootstrap.Toast(toastEl).show());
  }

  function initTableFilter({ rowSelector, searchSelector, filterSelector, emptySelector }) {
    const rows = Array.from(document.querySelectorAll(rowSelector));
    const search = document.querySelector(searchSelector);
    const filter = document.querySelector(filterSelector);
    const empty = document.querySelector(emptySelector);
    if (!rows.length || (!search && !filter)) return;

    const applyFilter = () => {
      const query = (search?.value || "").trim().toLowerCase();
      const selected = filter?.value || "all";
      let shown = 0;

      rows.forEach((row) => {
        const searchable = row.dataset.search || row.textContent.toLowerCase();
        const risk = row.dataset.risk || "";
        const label = row.dataset.label || "";
        const matchesQuery = !query || searchable.includes(query);
        const matchesFilter = selected === "all" || selected === risk || selected === label;
        const visible = matchesQuery && matchesFilter;
        row.classList.toggle("d-none", !visible);
        if (visible) shown += 1;
      });

      if (empty) empty.classList.toggle("d-none", shown !== 0);
    };

    search?.addEventListener("input", applyFilter);
    filter?.addEventListener("change", applyFilter);
  }



  function observeOnce(element, callback, options = { threshold: 0.25 }) {
    if (!element) return;
    if (!("IntersectionObserver" in window)) {
      callback();
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        callback();
        observer.disconnect();
      });
    }, options);
    observer.observe(element);
  }

  function animateCount(element) {
    if (!element || element.dataset.animated === "true") return;
    const target = Number.parseFloat(element.dataset.count || "0");
    if (!Number.isFinite(target)) return;

    const decimals = Number.parseInt(element.dataset.decimals ?? "2", 10);
    const suffix = element.dataset.suffix || "";
    const prefix = element.dataset.prefix || "";
    const duration = Number.parseInt(element.dataset.duration || "1350", 10);
    const start = performance.now();

    element.dataset.animated = "true";
    element.classList.add("is-counting");

    const easeOut = (t) => 1 - Math.pow(1 - t, 3);
    const formatValue = (value) => `${prefix}${value.toFixed(Math.max(decimals, 0))}${suffix}`;

    function frame(now) {
      const progress = Math.min((now - start) / duration, 1);
      const current = target * easeOut(progress);
      element.textContent = formatValue(current);
      if (progress < 1) {
        requestAnimationFrame(frame);
      } else {
        element.textContent = formatValue(target);
        element.classList.remove("is-counting");
      }
    }

    requestAnimationFrame(frame);
  }

  function initCountUps() {
    document.querySelectorAll(".count-up").forEach((element) => {
      observeOnce(element, () => animateCount(element), { threshold: 0.35 });
    });
  }

  function initProgressBars() {
    document.querySelectorAll(".animated-progress-bar").forEach((bar) => {
      observeOnce(bar, () => {
        const value = Number.parseFloat(bar.dataset.progress || "0");
        const safeValue = Number.isFinite(value) ? Math.max(0, Math.min(value, 100)) : 0;
        bar.style.width = `${safeValue}%`;
        bar.classList.add("is-animated");
      }, { threshold: 0.35 });
    });
  }


  document.addEventListener("DOMContentLoaded", () => {
    requestAnimationFrame(() => {
      root.classList.remove("theme-loading");
    });
    updateThemeButtons(savedTheme);
    initTooltips();
    initToasts();
    initCountUps();
    initProgressBars();

    const navbar = document.getElementById("mainNavbar");

    function updateNavbarOnScroll() {
      if (!navbar) return;
      navbar.classList.toggle("is-scrolled", window.scrollY > 24);
    }

    updateNavbarOnScroll();
    window.addEventListener("scroll", updateNavbarOnScroll, { passive: true });

    document.querySelectorAll(".theme-toggle").forEach((button) => {
      button.addEventListener("click", () => {
        const current = root.getAttribute("data-theme") || "light";
        const next = current === "dark" ? "light" : "dark";
        root.setAttribute("data-theme", next);
        localStorage.setItem("diabetes-ai-theme", next);
        updateThemeButtons(next);
      });
    });

    // Native <details> dropdown fallback: reliable click behavior even with glass layers.
    const navDetails = document.querySelectorAll(".nav-details");
    navDetails.forEach((details) => {
      details.addEventListener("toggle", () => {
        if (!details.open) return;
        navDetails.forEach((other) => {
          if (other !== details) other.removeAttribute("open");
        });
      });
    });

    document.addEventListener("click", (event) => {
      navDetails.forEach((details) => {
        if (!details.contains(event.target)) details.removeAttribute("open");
      });
    });

    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      navDetails.forEach((details) => details.removeAttribute("open"));
    });

    const sampleButtons = document.querySelectorAll(".sample-btn");
    sampleButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const key = button.dataset.sampleKey;
        const sample = window.samplePatients?.[key]?.values;
        if (!sample) return;

        Object.entries(sample).forEach(([field, value]) => {
          const input = document.getElementById(field);
          if (input) input.value = value;
        });

        sampleButtons.forEach((btn) => btn.classList.remove("active"));
        button.classList.add("active");
      });
    });

    const forms = document.querySelectorAll(".app-form");
    forms.forEach((form) => {
      form.addEventListener("submit", () => {
        const submitButton = form.querySelector(".submit-btn");
        if (!submitButton) return;
        const normal = submitButton.querySelector(".btn-normal");
        const loading = submitButton.querySelector(".btn-loading");
        submitButton.disabled = true;
        normal?.classList.add("d-none");
        loading?.classList.remove("d-none");
      });
    });

    initTableFilter({
      rowSelector: ".batch-result-row",
      searchSelector: "#batchSearch",
      filterSelector: "#batchRiskFilter",
      emptySelector: "#batchNoRows"
    });

    initTableFilter({
      rowSelector: ".history-row",
      searchSelector: "#historySearch",
      filterSelector: "#historyRiskFilter",
      emptySelector: "#historyNoRows"
    });

    const chartTextColor = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim() || "#64748b";

    if (window.batchCharts && window.Chart) {
      const riskCanvas = document.getElementById("riskChart");
      const predictionCanvas = document.getElementById("predictionChart");

      if (riskCanvas) {
        observeOnce(riskCanvas, () => {
          new Chart(riskCanvas, {
            type: "bar",
            data: {
              labels: window.batchCharts.risk.labels,
              datasets: [{
                label: "Records",
                data: window.batchCharts.risk.values,
                backgroundColor: ["#169296", "#027fbc", "#015877"],
                borderRadius: 12,
              }]
            },
            options: {
              responsive: true,
              animation: {
                duration: 1300,
                easing: "easeOutQuart",
                delay: (context) => context.type === "data" ? context.dataIndex * 110 : 0
              },
              plugins: { legend: { display: false } },
              scales: {
                x: { ticks: { color: chartTextColor }, grid: { display: false } },
                y: { ticks: { color: chartTextColor }, beginAtZero: true }
              }
            }
          });
        });
      }

      if (predictionCanvas) {
        observeOnce(predictionCanvas, () => {
          new Chart(predictionCanvas, {
            type: "doughnut",
            data: {
              labels: window.batchCharts.prediction.labels,
              datasets: [{
                data: window.batchCharts.prediction.values,
                backgroundColor: ["#169296", "#015877"],
                borderWidth: 0,
              }]
            },
            options: {
              responsive: true,
              cutout: "68%",
              animation: { animateRotate: true, animateScale: true, duration: 1400, easing: "easeOutQuart" },
              plugins: { legend: { position: "bottom", labels: { color: chartTextColor } } }
            }
          });
        });
      }
    }

    if (window.performanceCharts && window.Chart) {
      const featureCanvas = document.getElementById("featureImportanceChart");
      const thresholdCanvas = document.getElementById("thresholdChart");

      if (featureCanvas) {
        observeOnce(featureCanvas, () => {
          const featureLabels = window.performanceCharts.features.labels;
          const featureValues = window.performanceCharts.features.values.map(Number);
          const initialValues = featureValues.map(() => 0);

          const featureChart = new Chart(featureCanvas, {
            type: "bar",
            data: {
              labels: featureLabels,
              datasets: [{
                label: "Importance (%)",
                data: initialValues,
                backgroundColor: (context) => {
                  const chart = context.chart;
                  const { ctx, chartArea } = chart;
                  if (!chartArea) return "#20bab5";

                  const gradient = ctx.createLinearGradient(chartArea.left, 0, chartArea.right, 0);
                  gradient.addColorStop(0, "#015877");
                  gradient.addColorStop(0.45, "#169296");
                  gradient.addColorStop(1, "#20bab5");
                  return gradient;
                },
                borderRadius: 12,
                barPercentage: .72,
                categoryPercentage: .72,
              }]
            },
            options: {
              indexAxis: "y",
              responsive: true,
              maintainAspectRatio: false,
              animation: {
                duration: 1600,
                easing: "easeOutQuart"
              },
              animations: {
                x: {
                  from: 0,
                  duration: 1600,
                  easing: "easeOutQuart"
                },
                y: {
                  duration: 700,
                  easing: "easeOutCubic"
                }
              },
              plugins: {
                legend: { display: false },
                tooltip: {
                  callbacks: {
                    label: (context) => `Importance: ${context.raw}%`
                  }
                }
              },
              scales: {
                x: {
                  ticks: { color: chartTextColor },
                  beginAtZero: true,
                  grid: { color: "rgba(255,255,255,.05)" }
                },
                y: {
                  ticks: { color: chartTextColor },
                  grid: { display: false }
                }
              }
            }
          });

          setTimeout(() => {
            featureChart.data.datasets[0].data = featureValues;
            featureChart.update();
          }, 180);
        }, { threshold: 0.2 });
      }

      if (thresholdCanvas) {
        observeOnce(thresholdCanvas, () => {
          const thresholdLabels = window.performanceCharts.thresholds.labels;

          const f1Values = window.performanceCharts.thresholds.f1.map(Number);
          const recallValues = window.performanceCharts.thresholds.recall.map(Number);
          const precisionValues = window.performanceCharts.thresholds.precision.map(Number);

          const emptyLine = thresholdLabels.map(() => null);

          const thresholdChart = new Chart(thresholdCanvas, {
            type: "line",
            data: {
              labels: thresholdLabels,
              datasets: [
                {
                  label: "F1",
                  data: emptyLine,
                  borderColor: "#027fbc",
                  backgroundColor: "#027fbc",
                  tension: .38,
                  pointRadius: 4,
                  pointHoverRadius: 7,
                  borderWidth: 3
                },
                {
                  label: "Recall",
                  data: emptyLine,
                  borderColor: "#20bab5",
                  backgroundColor: "#20bab5",
                  tension: .38,
                  pointRadius: 4,
                  pointHoverRadius: 7,
                  borderWidth: 3
                },
                {
                  label: "Precision",
                  data: emptyLine,
                  borderColor: "#169296",
                  backgroundColor: "#169296",
                  tension: .38,
                  pointRadius: 4,
                  pointHoverRadius: 7,
                  borderWidth: 3
                }
              ]
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              animation: {
                duration: 1700,
                easing: "easeOutQuart"
              },
              animations: {
                y: {
                  from: 0.75,
                  duration: 1700,
                  easing: "easeOutQuart"
                },
                x: {
                  duration: 900,
                  easing: "easeOutCubic"
                }
              },
              plugins: {
                legend: {
                  position: "bottom",
                  labels: { color: chartTextColor }
                }
              },
              scales: {
                x: {
                  ticks: { color: chartTextColor },
                  grid: { display: false }
                },
                y: {
                  ticks: { color: chartTextColor },
                  min: 0.75,
                  max: 0.92,
                  grid: { color: "rgba(255,255,255,.05)" }
                }
              }
            }
          });

          setTimeout(() => {
            thresholdChart.data.datasets[0].data = f1Values;
            thresholdChart.data.datasets[1].data = recallValues;
            thresholdChart.data.datasets[2].data = precisionValues;
            thresholdChart.update();
          }, 220);
        }, { threshold: 0.25 });
      }
    }
  });
})();
