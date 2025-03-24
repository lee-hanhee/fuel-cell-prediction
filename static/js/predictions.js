/**
 * Handles prediction results display and visualization
 * Initializes result charts and manages prediction form submissions
 */

// Document ready function
document.addEventListener("DOMContentLoaded", function () {
  // Check if prediction results exist
  if (
    document.getElementById("pressure-result") &&
    document.getElementById("temperature-result")
  ) {
    createResultCharts();
  }

  // Add event listener to prediction form submission
  const predictionForm = document.getElementById("prediction-form");
  if (predictionForm) {
    predictionForm.addEventListener("submit", function () {
      // Show loading spinner
      const spinner = document.getElementById("loading-spinner");
      if (spinner) {
        spinner.classList.remove("d-none");
      }

      // Disable submit button during form submission
      const submitBtn = document.getElementById("predict-btn");
      if (submitBtn) {
        submitBtn.disabled = true;
      }
    });
  }
});

/**
 * Creates gauge charts to visualize the prediction results
 * Displays pressure drop and temperature values as gauge charts
 */
function createResultCharts() {
  // Get the pressure value from the result display
  const pressureElement = document.getElementById("pressure-result");
  const temperatureElement = document.getElementById("temperature-result");

  if (!pressureElement || !temperatureElement) return;

  // Parse the result values
  let pressureValue = parseFloat(pressureElement.textContent.trim());
  let temperatureValue = parseFloat(temperatureElement.textContent.trim());

  // Check if we have valid values
  if (isNaN(pressureValue) || isNaN(temperatureValue)) return;

  // Create container elements for charts
  const pressureChartContainer = document.createElement("div");
  pressureChartContainer.style.marginTop = "1rem";
  pressureChartContainer.style.height = "150px";
  pressureElement.parentNode.appendChild(pressureChartContainer);

  const temperatureChartContainer = document.createElement("div");
  temperatureChartContainer.style.marginTop = "1rem";
  temperatureChartContainer.style.height = "150px";
  temperatureElement.parentNode.appendChild(temperatureChartContainer);

  // Create gauge charts
  createPressureGauge(pressureChartContainer, pressureValue);
  createTemperatureGauge(temperatureChartContainer, temperatureValue);
}

/**
 * Creates a gauge chart for pressure drop visualization
 * @param {HTMLElement} container - The container element for the chart
 * @param {number} value - The pressure drop value to display
 */
function createPressureGauge(container, value) {
  // Determine the max value for the gauge scale
  const maxValue = Math.max(500, Math.ceil((value * 1.5) / 100) * 100);

  // Create the gauge chart using Plotly
  const data = [
    {
      type: "indicator",
      mode: "gauge+number",
      value: value,
      number: { suffix: " Pa", font: { size: 24 } },
      gauge: {
        axis: { range: [null, maxValue], tickwidth: 1 },
        bar: { color: "#1565C0" },
        bgcolor: "white",
        borderwidth: 2,
        bordercolor: "gray",
        steps: [
          { range: [0, maxValue * 0.33], color: "#E3F2FD" },
          { range: [maxValue * 0.33, maxValue * 0.66], color: "#90CAF9" },
          { range: [maxValue * 0.66, maxValue], color: "#42A5F5" },
        ],
        threshold: {
          line: { color: "red", width: 4 },
          thickness: 0.75,
          value: value,
        },
      },
    },
  ];

  const layout = {
    margin: { t: 25, r: 25, l: 25, b: 25 },
    font: { size: 12 },
  };

  Plotly.newPlot(container, data, layout, { responsive: true });
}

/**
 * Creates a gauge chart for stack temperature visualization
 * @param {HTMLElement} container - The container element for the chart
 * @param {number} value - The temperature value to display
 */
function createTemperatureGauge(container, value) {
  // Determine appropriate range for the temperature gauge
  const maxValue = Math.max(100, Math.ceil((value * 1.2) / 10) * 10);

  // Create the gauge chart using Plotly
  const data = [
    {
      type: "indicator",
      mode: "gauge+number",
      value: value,
      number: { suffix: " °C", font: { size: 24 } },
      gauge: {
        axis: { range: [null, maxValue], tickwidth: 1 },
        bar: { color: "#C62828" },
        bgcolor: "white",
        borderwidth: 2,
        bordercolor: "gray",
        steps: [
          { range: [0, maxValue * 0.33], color: "#FFEBEE" },
          { range: [maxValue * 0.33, maxValue * 0.66], color: "#FFCDD2" },
          { range: [maxValue * 0.66, maxValue], color: "#EF9A9A" },
        ],
        threshold: {
          line: { color: "red", width: 4 },
          thickness: 0.75,
          value: value,
        },
      },
    },
  ];

  const layout = {
    margin: { t: 25, r: 25, l: 25, b: 25 },
    font: { size: 12 },
  };

  Plotly.newPlot(container, data, layout, { responsive: true });
}

/**
 * Updates chart theme based on the current site theme
 * @param {boolean} isDarkMode - Whether the site is in dark mode
 */
function updateChartsTheme(isDarkMode) {
  // Get all plotly charts
  const chartElements = document.querySelectorAll('[id^="plotly-"]');

  chartElements.forEach(function (chartElement) {
    if (chartElement && chartElement.data) {
      const update = {
        paper_bgcolor: isDarkMode ? "#212529" : "white",
        "font.color": isDarkMode ? "white" : "black",
        "gauge.bgcolor": isDarkMode ? "#2d2d2d" : "white",
      };

      Plotly.relayout(chartElement, update);
    }
  });
}
