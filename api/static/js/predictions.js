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

// Create charts for prediction results
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

  // Create canvas elements
  const pressureCanvas = document.createElement("canvas");
  pressureCanvas.id = "pressure-chart";
  pressureChartContainer.appendChild(pressureCanvas);

  const temperatureCanvas = document.createElement("canvas");
  temperatureCanvas.id = "temperature-chart";
  temperatureChartContainer.appendChild(temperatureCanvas);

  // Create gauge chart for pressure
  createGaugeChart(
    pressureCanvas,
    "Pressure Drop (Pa)",
    pressureValue,
    0,
    100,
    "#0d6efd"
  );

  // Create gauge chart for temperature
  createGaugeChart(
    temperatureCanvas,
    "Stack Temperature (°C)",
    temperatureValue,
    0,
    100,
    "#dc3545"
  );
}

// Create a gauge chart
function createGaugeChart(canvas, label, value, min, max, color) {
  // Calculate percentage for the gauge
  const percent = ((value - min) / (max - min)) * 100;

  // Create chart
  new Chart(canvas, {
    type: "doughnut",
    data: {
      datasets: [
        {
          data: [percent, 100 - percent],
          backgroundColor: [color, "#e9ecef"],
          borderWidth: 0,
        },
      ],
    },
    options: {
      cutout: "70%",
      circumference: 180,
      rotation: -90,
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        animateRotate: true,
        animateScale: true,
      },
      plugins: {
        tooltip: {
          enabled: false,
        },
        legend: {
          display: false,
        },
        title: {
          display: true,
          text: value.toFixed(2),
          position: "bottom",
          font: {
            size: 18,
            weight: "bold",
          },
        },
        subtitle: {
          display: true,
          text: label,
          position: "bottom",
          font: {
            size: 14,
          },
        },
      },
    },
  });
}

// Function to update the theme of any charts when dark mode is toggled
function updateChartsTheme(isDarkMode) {
  // Get all canvas elements for charts
  const chartElements = document.querySelectorAll('canvas[id$="-chart"]');

  chartElements.forEach((element) => {
    const chart = Chart.getChart(element);
    if (chart) {
      // Update chart theme
      chart.options.plugins.title.color = isDarkMode ? "#ffffff" : "#212529";
      chart.options.plugins.subtitle.color = isDarkMode ? "#cccccc" : "#6c757d";

      // Update the background color of the empty part of the gauge
      chart.data.datasets[0].backgroundColor[1] = isDarkMode
        ? "#444444"
        : "#e9ecef";

      // Update the chart
      chart.update();
    }
  });
}
